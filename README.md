# Align Affordance

This codebase is based on https://github.com/NVlabs/affordance_diffusion. Please follow the instructions there for data generation and setup.

## Reward Finetuning
Please see `configs/reward_finetune.yaml` for additional hyperparameters. To run the reward finetuning, run the following command:

```bash
python reward_finetune.py data.data_dir='/path/to/all_processed_data/HOI4D_glide/denoised_obj/*.*g' what_ckpt=/path/to/content_ldm/checkpoints/last.ckpt test_name=<name-of-run> detect_reward_scale=10.0 ambig_reward_scale=20.0
```

## Evaluation
To evaluate the model, run the following command:

```bash
python inference.py data.data_dir='/path/to/eval-data/obj_cls_only/*.*g' test_num=10 metric=fid dir=/path/to/eval-results/ dirB=/path/to/eval-data/corrsp_hoi_only/ what_ckpt=/path/to/content_ldm/checkpoints/last.ckpt align_ckpt=/path/to/my-finetuned.ckpt
```