# wild mix of https://github.com/openai/glide-text2im and https://github.com/ermongroup/SDEdit
# MIT License

# Copyright (c) 2021 OpenAI https://github.com/openai/glide-text2im
# Copyright (c) 2021 Ermon Group https://github.com/ermongroup/SDEdit

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# --------------------------------------------------------
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
# see CC-BY-NC-SA-4.0.md for details
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------

from argparse import ArgumentParser
import os
import os.path as osp
from typing import Tuple
from tqdm import tqdm
import pandas
import json
import pickle
from PIL import Image
import cv2
import numpy as np
import time
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.dataloader import default_collate


from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)

from sdedit import denoise_base, load_denoise_base
from sdedit import upsample as denoised_up
from sdedit import load_up as load_denoise_up

from dataset import HOI4D


has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')


def image2np(batch: th.Tensor):
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3]).detach().numpy()
    return reshaped


def read_image(path: str, size: int = 256) -> Tuple[th.Tensor, th.Tensor]:
    pil_img = Image.open(path).convert('RGB')
    pil_img = pil_img.resize((size, size), resample=Image.BICUBIC)
    img = np.array(pil_img)
    return th.from_numpy(img)[None].permute(0, 3, 1, 2).float() / 127.5 - 1


def load_base():
    # Create base model.
    options = model_and_diffusion_defaults()
    options['inpaint'] = True
    options['use_fp16'] = has_cuda
    options['timestep_respacing'] = '100' # use 100 diffusion steps for fast sampling
    model, diffusion = create_model_and_diffusion(**options)
    model.eval()
    if has_cuda:
        model.convert_to_fp16()
    model.to(device)
    if args.base_ckpt is None:
        model.load_state_dict(load_checkpoint('base-inpaint', device))
    else:
        model.load_state_dict(th.load(args.base_ckpt))
    print('total base parameters', sum(x.numel() for x in model.parameters()))
    return model, diffusion, options


def load_up():
    # Create upsampler model.
    options_up = model_and_diffusion_defaults_upsampler()
    options_up['inpaint'] = True
    options_up['use_fp16'] = has_cuda
    options_up['timestep_respacing'] = 'fast27' # use 27 diffusion steps for very fast sampling
    model_up, diffusion_up = create_model_and_diffusion(**options_up)
    model_up.eval()
    if has_cuda:
        model_up.convert_to_fp16()
    model_up.to(device)
    model_up.load_state_dict(load_checkpoint('upsample-inpaint', device))
    print('total upsampler parameters', sum(x.numel() for x in model_up.parameters()))
    return model_up, diffusion_up, options_up


def predict_human(s=None,mask_file=None, verbose=True):
    if not osp.exists(mask_file):
        # deprecated: hand detection
        pass
    else:
        if verbose:
            print('exist hand make', mask_file)
        ori_human_masks = cv2.imread(mask_file)
        masks = (ori_human_masks > 122.5)[..., 0]

    masks = Image.fromarray(masks.astype(np.uint8))
    masks = masks.resize((256, 256), resample=Image.BICUBIC)
    
    # Creating kernel
    if s is not None:
        kernel = np.ones((s, s), np.uint8)

        # Using cv2.erode() method 
        masks = cv2.erode(np.array(masks), kernel)

        masks = th.FloatTensor(masks[None, None])
    else:
        masks = None
    return masks, ori_human_masks


def load_input(fname, mask_file=None, verbose=True):
    # Source image we are inpainting
    pil_img = Image.open(fname).convert('RGB')
    source_image_256 = read_image(fname, size=256)
    source_image_64 = read_image(fname, size=64)

    source_mask_256, orig_mask = predict_human(args.kernel, mask_file=mask_file, verbose=verbose)
    source_mask_64 = F.interpolate(source_mask_256, (64, 64), mode='nearest')    
    return source_image_256, source_mask_256, source_image_64, source_mask_64, pil_img, orig_mask


##############################
# Sample from the base model #
##############################
def base_generate(model, diffusion, options, source_image_64, source_mask_64, prompt, ):
    batch_size = source_image_64.shape[0]
    guidance_scale = args.scale # 5.0
    if args.verbose:    
        print('prompt: %s with scale %f' % (prompt, guidance_scale))

    # Create an classifier-free guidance sampling function
    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        return th.cat([eps, rest], dim=1)

    def denoised_fn(x_start):
        # Force the model to have the exact right x_start predictions
        # for the part of the image which is known.
        return (
            x_start * (1 - model_kwargs['inpaint_mask'])
            + model_kwargs['inpaint_image'] * model_kwargs['inpaint_mask']
        )

    # Sampling parameters
    # Create the text tokens to feed to the model.
    if isinstance(prompt, list):
        prompts = prompt
    else:
        prompts = [prompt]

    tokens = [model.tokenizer.encode(prompt) for prompt in prompts]
    tokens_and_masks = [model.tokenizer.padded_tokens_and_mask(t, options['text_ctx']) for t in tokens]
    tokens = [t for t, _ in tokens_and_masks]
    masks = [m for _, m in tokens_and_masks]

    # Create the classifier-free guidance tokens (empty)
    full_batch_size = batch_size * 2
    uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
        [], options['text_ctx']
    )

    # Pack the tokens together into model kwargs.
    model_kwargs = dict(
        tokens=th.tensor(
            tokens + [uncond_tokens] * batch_size, device=device
        ),
        mask=th.tensor(
            masks + [uncond_mask] * batch_size,
            dtype=th.bool,
            device=device,
        ),

        # Masked inpainting image
        inpaint_image=(source_image_64 * source_mask_64).repeat(2, 1, 1, 1).to(device),
        inpaint_mask=source_mask_64.repeat(2, 1, 1, 1).to(device),
    )

    # Sample from the base model.
    samples = diffusion.p_sample_loop(
        model_fn,
        (full_batch_size, 3, options["image_size"], options["image_size"]),
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
        denoised_fn=denoised_fn,
    )[:batch_size]

    # Show the output
    # show_images(samples, '%s_outputx64' % index)
    return samples


##############################
# Upsample the 64x64 samples #
##############################
def upsample(model_up, diffusion_up, options_up, samples, source_image_256, source_mask_256, prompt, ):
    batch_size = source_image_256.shape[0]
    upsample_temp = args.temp
    guidance_scale = args.scale

    if isinstance(prompt, list):
        prompts = prompt
    else:
        prompts = [prompt]

    tokens = [model_up.tokenizer.encode(prompt) for prompt in prompts]
    tokens_and_masks = [model_up.tokenizer.padded_tokens_and_mask(t, options_up['text_ctx']) for t in tokens]
    tokens = [t for t, _ in tokens_and_masks]
    masks = [m for _, m in tokens_and_masks]

    # Create the model conditioning dict.
    model_kwargs = dict(
        # Low-res image to upsample.
        low_res=((samples+1)*127.5).round()/127.5 - 1,

        # Text tokens
        tokens=th.tensor(
            tokens, device=device
        ),
        mask=th.tensor(
            masks,
            dtype=th.bool,
            device=device,
        ),

        # Masked inpainting image.
        inpaint_image=(source_image_256 * source_mask_256).repeat(1, 1, 1, 1).to(device),
        inpaint_mask=source_mask_256.repeat(1, 1, 1, 1).to(device),
    )

    def denoised_fn(x_start):
        # Force the model to have the exact right x_start predictions
        # for the part of the image which is known.
        return (
            x_start * (1 - model_kwargs['inpaint_mask'])
            + model_kwargs['inpaint_image'] * model_kwargs['inpaint_mask']
        )

    # Sample from the base model.
    up_shape = (batch_size, 3, options_up["image_size"], options_up["image_size"])
    up_samples = diffusion_up.p_sample_loop(
        model_up,
        up_shape,
        noise=th.randn(up_shape, device=device) * upsample_temp,
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
        denoised_fn=denoised_fn,
    )[:batch_size]

    return up_samples

def collate_w_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)

def dataloader(args):
    shuffle = True
    ds_list = []
    data_dir = args.data_dir
    save_dir = args.save_dir

    for split in args.split.split(','):
        ds = HOI4D(data_dir, save_dir, split, args.save_index)
        ds_list.append(ds)
    ds = ConcatDataset(ds_list)

    th.manual_seed(303)
    np.random.seed(303)

    batch_size = args.bs if args.bs > 1 else None
    shuffle = True
    dl = DataLoader(ds, batch_size, shuffle, num_workers=args.num_workers, drop_last=False, collate_fn=collate_w_none)

    return dl


def batch_main(args):
    dl = dataloader(args)

    glide = {}
    glide['denoise'] = load_denoise_base()
    glide['denoise_up'] = load_denoise_up()
    glide['base'] = load_base()
    glide['up'] = load_up()
    
    for i, data in tqdm(enumerate(dl), total=len(dl)):
        if data is None:
            continue
        if args.num > 0 and i > args.num:
            break
        if args.base_ckpt is not None:
            data['out_file'] = [osp.join(args.save_dir, osp.basename(data['out_file'][0]))]
        if args.skip and osp.exists(data['out_file']):
            print('skip', osp.exists(data['out_file']), data['out_file'])
            continue
        lock_file = data['out_file'] + '.lock'
        try:
            os.makedirs(lock_file)
        except FileExistsError:
            if args.skip:
                continue
        try:
            inpaint_image(data['inp_file'], data['out_file'], glide, data)  # shall we do crop first?
        except Exception as e:
            if not KeyboardInterrupt and not args.debug:
                continue
            print(e)
            raise e
        os.system('rm -r %s' % lock_file)
        print('rm ', lock_file)


def batch_main_parallel(args):
    dl = dataloader(args)

    glide = {}
    glide['denoise'] = load_denoise_base()
    glide['denoise_up'] = load_denoise_up()
    glide['base'] = load_base()
    glide['up'] = load_up()
    
    for i, data in tqdm(enumerate(dl), total=len(dl)):
        if data is None:
            continue
        if args.num > 0 and i > args.num:
            break
        if args.base_ckpt is not None:
            data['out_file'] = [osp.join(args.save_dir, osp.basename(data['out_file'][0]))]
        if args.skip and osp.exists(data['out_file']):
            print('skip', osp.exists(data['out_file']), data['out_file'])
            continue
        
        # TODO: better way to handle lock files
        lock_files = [out_file + '.lock' for out_file in data['out_file']]
        for lock_file in lock_files:
            try:
                os.makedirs(lock_file)
            except FileExistsError:
                if args.skip:
                    continue
        
        try:
            if args.verbose:
                print('inpainting')
            inpaint_image_parallel(data['inp_file'], data['out_file'], glide, data)  # shall we do crop first?
        except Exception as e:
            if not KeyboardInterrupt and not args.debug:
                continue
            print(e)
            raise e
        
        for lock_file in lock_files:
            os.system('rm -r %s' % lock_file)
            if args.verbose: 
                print('rm ', lock_file)

@th.no_grad()
def inpaint_image(inp_file, out_file, glide, data):
    prompt = data['prompt']

    source_image_256, source_mask_256, source_image_64, source_mask_64, ori_image, ori_mask \
        = load_input(inp_file, mask_file=data['mask_file'])

    if not args.dry:
        sample = base_generate(*glide['base'], source_image_64, source_mask_64, prompt)
        samplex256 = upsample(*glide['up'], sample, source_image_256, source_mask_256, prompt)

        out_image = Image.fromarray(image2np(samplex256))
        out_image = out_image.resize(ori_image.size)
    
        # save human mask as well~~
        if not osp.exists(data['mask_file']):
            os.makedirs(osp.dirname(data['mask_file']), exist_ok=True)
            print(data['mask_file'])
            ori_mask.save(data['mask_file'])

        os.makedirs(osp.dirname(out_file), exist_ok=True)
        print('save to ', out_file)
        out_image.save(out_file)

        inp_image = F.avg_pool2d(samplex256, 4)
        out_image64 = denoise_base(0.05, *glide['denoise'], inp_image, prompt)
        out_image = denoised_up(0, *glide['denoise_up'], out_image64, prompt)

        out_image = Image.fromarray(image2np(out_image))
        out_image = out_image.resize(ori_image.size)
        denoise_file = out_file.replace('glide_obj', 'denoised_obj')
        os.makedirs(osp.dirname(denoise_file), exist_ok=True)
        print(denoise_file)
        out_image.save(denoise_file)

@th.no_grad()
def inpaint_image_parallel(inp_files, out_files, glide, data):
    """For now, naively looping through loading of input and writing images.
    But we definitely need to parallelize the diffusion process across images.
    """
    prompts = data['prompt']
    mask_files = data['mask_file']

    # TODO: parallelize loading inputs
    source_images_256, source_masks_256, source_images_64, source_masks_64, ori_images, ori_masks = load_input_parallel(inp_files, mask_files)
    
    if not args.dry:
        sample = base_generate(*glide['base'], source_images_64, source_masks_64, prompts)
        samples_x256 = upsample(*glide['up'], sample, source_images_256, source_masks_256, prompts)

        # TODO: parallize saving inpaint mask images
        _save_inpaint_mask_parallel(samples_x256, ori_images, ori_masks, out_files, mask_files)

        inp_images = F.avg_pool2d(samples_x256, 4)
        out_images_64 = denoise_base(0.05, *glide['denoise'], inp_images, prompts)
        out_images = denoised_up(0, *glide['denoise_up'], out_images_64, prompts)

        # TODO: parallize saving denoised images
        _save_denoised_parallel(out_images, ori_images, out_files)


def load_input_parallel(inp_files, mask_files):
    """TODO: parallelize loading inputs"""

    source_images_256, source_masks_256, source_images_64, source_masks_64, ori_images, ori_masks = [], [], [], [], [], []
    for i, inp_file in enumerate(inp_files):
        source_image_256, source_mask_256, source_image_64, source_mask_64, ori_image, ori_mask \
            = load_input(inp_file, mask_file=mask_files[i], verbose=False)
        
        source_images_256.append(source_image_256)
        source_masks_256.append(source_mask_256)
        source_images_64.append(source_image_64)
        source_masks_64.append(source_mask_64)
        ori_images.append(ori_image)
        ori_masks.append(ori_mask)

    source_images_256 = th.cat(source_images_256, dim=0)
    source_masks_256 = th.cat(source_masks_256, dim=0)
    source_images_64 = th.cat(source_images_64, dim=0)
    source_masks_64 = th.cat(source_masks_64, dim=0)

    return source_images_256, source_masks_256, source_images_64, source_masks_64, ori_images, ori_masks

def _save_inpaint_mask_parallel(samples_x256, ori_images, ori_masks, out_files, mask_files):
    """TODO: parallelize saving images"""
    
    for samplex256, ori_image, ori_mask, mask_file, out_file in zip(samples_x256, ori_images, ori_masks, mask_files, out_files):
        out_image = Image.fromarray(image2np(samplex256.unsqueeze(0)))
        out_image = out_image.resize(ori_image.size)

        # save human mask as well~~
        if not osp.exists(mask_file):
            os.makedirs(osp.dirname(mask_file), exist_ok=True)
            print(mask_file)
            ori_mask.save(mask_file)

        os.makedirs(osp.dirname(out_file), exist_ok=True)
        out_image.save(out_file)    

def _save_denoised_parallel(out_images, ori_images, out_files):
    """TODO: parallelize saving images"""

    for out_image, ori_image, out_file in zip(out_images, ori_images, out_files):
        out_image = Image.fromarray(image2np(out_image.unsqueeze(0)))
        out_image = out_image.resize(ori_image.size)
        denoise_file = out_file.replace('glide_obj', 'denoised_obj')
        os.makedirs(osp.dirname(denoise_file), exist_ok=True)
        out_image.save(denoise_file)

def decode_one_vid(vid, image_dir):
    vid_dir = osp.join(image_dir, '{}', 'align_rgb/image.mp4')
    save_dir = osp.join(image_dir, '{}', 'align_frames')
    os.makedirs(save_dir.format(vid), exist_ok=True)

    cmd = 'rm -r %s/*' % save_dir.format(vid)
    print(cmd)
    os.system(cmd)
    
    cmd = "ffmpeg -hide_banner -loglevel fatal -i {} {}/%04d.png".format(vid_dir.format(vid), save_dir.format(vid))
    print(cmd)
    os.system(cmd) 


def decode_frame(csv_file, vid_index=None, rewrite=False, open_close_art_only=False):
    if vid_index is None:
        df = pandas.read_csv(csv_file)
        vid_index = list(set(df['vid_index']))
        
        if open_close_art_only:
            print('Only including videos with open/close actions on safes and storage furniture')

            art_classes = ["Safe", "StorageFurniture"]
            art_actions = ["open", "close"]

            art_df = df[df['class'].isin(art_classes) & df['action'].isin(art_actions)]
            vid_index = list(set(art_df['vid_index']))

    save_dir = osp.join(args.data_dir,  'HOI4D_release', '{}', 'align_frames')
    for vid in tqdm(vid_index):
        if rewrite:
            os.system('rm -r %s' % save_dir.format(vid))
            decode_one_vid(vid, osp.join(args.data_dir, 'HOI4D_release'))
            print('rewrite')
        else:
            try:
                Image.open(save_dir.format(vid) + '/0001.png')
                print('continue', save_dir.format(vid), csv_file)
                continue
            except:
                print(save_dir.format(vid) + '/0001.png')
                lock_file = save_dir.format(vid) + '.lock'
                try:
                    os.makedirs(lock_file)
                except FileExistsError:
                    continue
                os.system('rm -r %s/*' % save_dir.format(vid))
                decode_one_vid(vid, osp.join(args.data_dir, 'HOI4D_release'))
                os.system('rm -r %s' % lock_file)


def make_bbox(df_file):
#   {"image_path": "xxx.jpg", "hand_bbox_list":[{"left_hand":[x,y,w,h], "right_hand":[x,y,w,h]}], "body_bbox_list":[[x,y,w,h]]}
# Note that bbox format is [minX,minY,width,height]
    data_dir = args.data_dir
    vis_dir = osp.join(data_dir, 'vis/{}_{}.png')
    os.makedirs(vis_dir, exist_ok=True)
    image_dir = osp.join(data_dir, 'HOI4D_release/{}/align_frames/{:04d}.png')
    bbox_dir = osp.join(args.save_dir, 'HOI4D_glide/det_hand/{}.json')
    os.makedirs(bbox_dir, exist_ok=True)
    right_bbox_dir = osp.join(data_dir, 'handpose/refinehandpose_right/{}/{:d}.pickle')
    left_bbox_dir = osp.join(data_dir, 'handpose/refinehandpose_left/{}/{:d}.pickle')

    df = pandas.read_csv(df_file)
    for i, data in tqdm(df.iterrows(), total=len(df)):
        vid, fnum = data['vid_index'], data['frame_number']
        index_str = '{}_frame{:04d}'.format(data['vid_index'].replace('/', '_'), data['frame_number'])
        bbox_file = bbox_dir.format(index_str)
        if args.skip and osp.exists(bbox_file):
            continue
        lock_file = bbox_file + '.lock'
        try:
             os.makedirs(lock_file)
        except FileExistsError:
            if args.skip:
                continue

        def get_bbox_xywh(obj_file):
            with open(obj_file, 'rb') as fp:
                obj = pickle.load(fp)
            kpts = obj['kps2D']
            x1 = int(min(kpts[:, 0]))
            y1 = int(min(kpts[:, 1]))
            x2 = int(max(kpts[:, 0]))
            y2 = int(max(kpts[:, 1]))
            return [x1, y1, x2-x1, y2-y1]

        hand_obj = {}
        if osp.exists(right_bbox_dir.format(vid, fnum)):
            hand_obj['right_hand'] = get_bbox_xywh(right_bbox_dir.format(vid, fnum))
        if osp.exists(left_bbox_dir.format(vid, fnum)):
            hand_obj['left_hand'] = get_bbox_xywh(left_bbox_dir.format(vid, fnum))
        bbox_obj = {'image_path': index_str, 'hand_bbox_list': [hand_obj], 'body_bbox_list': [[0, 0, 100, 100]]}
        with open(bbox_file, 'w') as fp:
            json.dump(bbox_obj, fp)
        os.system('rm -r %s' % lock_file)
    return 


def parser_args():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/hoi4d/')
    parser.add_argument('--save_dir', type=str, default='output/tmp_hoi4d/')
    parser.add_argument('--save_index', type=str, default='HOI4D_glide')
    parser.add_argument('--split', type=str, default='docs/all_contact.csv')
    
    parser.add_argument('--base_ckpt', type=str, default=None)
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--scale', type=float, default=5)
    parser.add_argument('--kernel', type=int, default=7)
    parser.add_argument('--temp', type=float, default=0.997)

    parser.add_argument('--num', type=int, default=-1)

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--skip', action='store_true')
    parser.add_argument('--dry', action='store_true')
    parser.add_argument('--decode', action='store_true')
    parser.add_argument('--inpaint', action='store_true')
    parser.add_argument('--bbox', action='store_true')
    parser.add_argument('--dir', type=str)
    parser.add_argument('--verbose', action='store_true')

    parser.add_argument('--open_close_art_only', action='store_true')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parser_args()
    print('save to', args.save_dir )
    if args.decode:
        print('#'*20, f'\n### Decoding frames from {args.split} ###\n', '#'*20)
        decode_frame(args.split, open_close_art_only=args.open_close_art_only)
    if args.inpaint:
        print('#'*20, f'\n### Inpainting frames from {args.split} ###\n', '#'*20) 
        if args.bs == 1:
            batch_main(args)
        else:
            batch_main_parallel(args)
    if args.bbox:
        print('#'*20, f'\n### Making bbox from {args.split} ###\n', '#'*20)
        make_bbox(args.split)
