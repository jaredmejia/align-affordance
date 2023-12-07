import pickle
from glob import glob
import random
import cv2
import imageio
import numpy as np
import json
import os
import os.path as osp
from hydra import main
import hydra.utils as hydra_utils
import pandas
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from models.hand_proxy import build_stn
from utils.train_utils import load_from_checkpoint
from jutils import image_utils
from dataset.dataset import build_imglist_dataloader
from utils import overlay_hand
import torchvision
from PIL import Image

import sys
sys.path.insert(0, 'third_party/frankmocap')

from inference import CascadeAffordance
from third_party.frankmocap.handmocap.hand_bbox_detector import Ego_Centric_Detector
from third_party.frankmocap.detectors.hand_object_detector.lib.model.rpn.bbox_transform import bbox_transform_inv
from third_party.frankmocap.detectors.hand_object_detector.lib.model.rpn.bbox_transform import clip_boxes



class AlignLoss(nn.Module):
    def __init__(self, ambig_reward_scale=1.0, detect_reward_scale=1.0, ambig_reward_type='mean', detect_reward_type='sum'):
        super().__init__()

        if ambig_reward_scale < 0 or detect_reward_scale < 0:
            raise ValueError("reward scale must be non-negative")
        
        if ambig_reward_type not in ['mean', 'top']:
            raise ValueError("ambig_reward_type must be 'mean' or 'top'")
        
        if detect_reward_type not in ['sum', 'mean', 'top']:
            raise ValueError("detect_reward_type must be 'sum', 'mean', or 'top'")
        
        self.ambig_reward_scale = ambig_reward_scale
        self.detect_reward_scale = detect_reward_scale
        self.ambig_reward_type = ambig_reward_type
        self.detect_reward_type = detect_reward_type
        self.detector = Ego_Centric_Detector()
        self.detector.hand_detector.eval()

    def reward_fn(self, ambig_score, detect_score):
        ambig_reward = self.ambig_reward_scale * (-0.5 + 2 * (ambig_score - 0.5) ** 2)
        detect_reward = (self.detect_reward_scale ** 2) * detect_score
        reward = ambig_reward + detect_reward
        return reward, ambig_reward.detach(), detect_reward.detach()
    
    def get_image_blob(self, img):
        orig_dims = img.shape[:2]
        img = torch.nn.Upsample(size=(600, 600), mode='linear')(img)
        scale_factor = 600.0 / min(orig_dims)
        
    
    def get_reward_components(self, img):
        """Get loss components for a single image.
        
        Based on frankamocap/handmocap/hand_bbox_detector.py
        """
        num_boxes = torch.LongTensor(1).cuda()
        gt_boxes = torch.FloatTensor(1).cuda()
        box_info = torch.FloatTensor(1).cuda()

        # scale image
        img = img / 2 + 0.5
        img = img * 255.0
        img = torch.clamp(img, 0, 255)
        img = img[[2, 1, 0], :, :] # RGB -> BGR to match Ego_Centric_Detector format
        pixel_means = torch.tensor([102.9801, 115.9465, 122.7717])[:, None, None].cuda()
        img = img - pixel_means

        orig_dim = img.shape[1:]
        im_data = torch.nn.Upsample(size=(600, 600), mode='bilinear')(img.unsqueeze(0))
        im_scale = 600.0/min(orig_dim)
        im_info = torch.tensor([600.0, 600.0, im_scale]).cuda().unsqueeze(0)

        gt_boxes.resize_(1, 1, 5).zero_()
        num_boxes.resize_(1).zero_()
        box_info.resize_(1, 1, 5).zero_() 

        # forward
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, loss_list = self.detector.hand_detector(im_data, im_info, gt_boxes, num_boxes, box_info) 

        scores = cls_prob
        boxes = rois.data[:, :, 1:5]

        # hand side info (left/right)
        lr_vector = loss_list[2][0]
        lr = torch.sigmoid(lr_vector) > 0.5
        lr = lr.squeeze(0).float()

        box_deltas = bbox_pred.data
        stds = [0.1, 0.1, 0.2, 0.2]
        means = [0.0, 0.0, 0.0, 0.0]
        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(stds).cuda() \
            + torch.FloatTensor(means).cuda()
        box_deltas = box_deltas.view(1, -1, 4 * len(self.detector.classes))

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

        pred_boxes /= im_scale
        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()

        thresh_hand = 0.5
        j = 2
        inds = torch.nonzero(scores[:, j]>thresh_hand, as_tuple=False).view(-1)

        if inds.numel() > 0:
            cls_boxes = pred_boxes[inds][:, j*4 : (j+1)*4]
            cls_scores = scores[:,j][inds]

            _, order = torch.sort(cls_scores, 0, True)

            lr = lr[inds][order]
            cls_boxes = cls_boxes[order]
            cls_scores = cls_scores[order]            
            lr_scores = torch.sigmoid(lr_vector).squeeze(0)[inds][order]

            if self.ambig_reward_type == 'mean':
                ambig_score = lr_scores.mean()
            else:
                ambig_score = lr_scores[0]

            if self.detect_reward_type == 'mean':
                detect_score = cls_scores.mean()
            elif self.detect_reward_type == 'top':
                detect_score = cls_scores[0]
            else:
                detect_score = cls_scores.sum()
        
        else:
            ambig_score = None
            detect_score = None

        return ambig_score, detect_score

    def forward_single(self, img):
        ambig_score, detect_score = self.get_reward_components(img)
        
        if ambig_score is None or detect_score is None:
            return None
        
        reward, ambig_reward, detect_reward = self.reward_fn(ambig_score, detect_score)
        loss = -1.0 * reward
        return loss, ambig_reward, detect_reward
    
    def forward(self, imgs):
        losses = []
        ambig_rewards = []
        detect_rewards = []
        for img in imgs:
            loss_result = self.forward_single(img)
            if loss_result is None:
                continue

            loss, ambig_reward, detect_reward = loss_result
            losses.append(loss)
            ambig_rewards.append(ambig_reward)
            detect_rewards.append(detect_reward)

        if len(losses) == 0:
            return None

        losses = torch.stack(losses)
        ambig_rewards = torch.stack(ambig_rewards)
        detect_rewards = torch.stack(detect_rewards)

        return losses.mean(), ambig_rewards.mean(), detect_rewards.mean()


class AlignAffordance(pl.LightningModule):
    def __init__(self, ambig_reward_scale, detect_reward_scale, ambig_reward_type, detect_reward_type, aff_model_args, aff_model_kwargs, num_images_log=4, image_log_freq=10):
        super().__init__()
        self.aff_model = CascadeAffordance(*aff_model_args, **aff_model_kwargs)
        
        self.cfg = self.aff_model.cfg
        self.cfg.dir = self.aff_model.save_dir
        self.cfg.use_flip = False
        
        # initialize layout net and set content net for training
        self.aff_model.init_model()
        self.content_net = self.aff_model.what
        self.content_net.train()

        if self.aff_model.what_cfg is not None:
            self.cfg.side_x = self.aff_model.what_cfg.side_x
        if self.aff_model.where_cfg is not None:
            self.cfg.side_x = max(self.cfg.side_x, self.aff_model.where_cfg.side_x)
        self.cfg.side_x = 256

        self.align_loss = AlignLoss(ambig_reward_scale, detect_reward_scale, ambig_reward_type, detect_reward_type)

        self.num_images_log = num_images_log
        self.image_log_freq = image_log_freq

    def training_step(self, batch, batch_idx):
        tokens, masks, reals, inpaint_image, inpaint_mask, mask_param, _ = batch

        # keep layout net frozen during finetuning
        self.aff_model.where.eval()
        for module in self.aff_model.where.modules():
            for param in module.parameters():
                param.requires_grad = False
        
        # randomized truncation backpropagation
        if self.cfg.randomized_truncation:
            num_steps_backprop = random.randrange(1, self.cfg.max_num_steps_backprop + 1)
        else:
            num_steps_backprop = self.cfg.max_num_steps_backprop

        samples, mask_param, inpaint_mask, = self.aff_model(batch, keep_grad=True, num_steps_backprop=num_steps_backprop)

        # log images
        if self.global_step % self.image_log_freq == 0:
            gen_images = samples[:self.num_images_log].clone().detach().cpu()
            gen_images = [image for image in gen_images]
            self.logger.log_image('gen_images', gen_images, self.global_step)

        # compute loss
        self.align_loss.detector.hand_detector.eval()
        loss_result = self.align_loss(samples)
        
        if loss_result is None:
            return None
        
        loss, ambig_reward, detect_reward = loss_result

        self.log('train_loss', loss.clone().detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        metrics = {
            'train_loss': loss.clone().detach(),
            'ambig_reward': ambig_reward.clone().detach(),
            'detect_reward': detect_reward.clone().detach(),
        }
        self.logger.log_metrics(metrics, self.global_step)

        return loss

    def forward(self, batch):
        return self.aff_model(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [x for x in self.content_net.glide_model.parameters() if x.requires_grad],
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.adam_weight_decay,
        )
        return optimizer

        
@main('configs', 'finetune', version_base=None)   
def main_worker(cfg):
    pl.seed_everything(47)

    aff_model_args = (cfg.what_ckpt, cfg.where_ckpt)
    aff_model_kwargs = {
        "test_name": cfg.test_name,
        "S": cfg.test_num,
        "cfg": cfg,
        "save_dir": cfg.dir,
    }
    align_affordance = AlignAffordance(cfg.ambig_reward_scale, cfg.detect_reward_scale, cfg.ambig_reward_type, cfg.detect_reward_type, aff_model_args, aff_model_kwargs, cfg.num_images_log, cfg.image_log_freq)

    dataloader = build_imglist_dataloader(cfg, cfg.data.data_dir, cfg.data.split, True, cfg.batch_size, shuffle=True)
    
    logger = WandbLogger(project='align-affordance', name=cfg.test_name, entity="jaredmejia", config=cfg)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=cfg.save_ckpt_dir,
        filename='model-{step:02d}-{train_loss:.2f}',
        monitor='train_loss',
        mode='min',
        save_top_k=5,
        save_last=True,
        save_weights_only=True,
        verbose=True,
        every_n_train_steps=cfg.save_ckpt_freq,
    )
    trainer = pl.Trainer(
        gpus=-1,
        max_steps=cfg.max_steps,
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(align_affordance, train_dataloaders=dataloader)


if __name__ == '__main__':
    main_worker()
