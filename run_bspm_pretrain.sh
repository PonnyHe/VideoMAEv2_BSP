#!/bin/bash

GPU_ID=${1:-0}   # 默认 GPU=0，可在命令行传参

CUDA_VISIBLE_DEVICES=$GPU_ID python main_pretrain.py \
  --config bspm_vitb_patch522.yaml \
  --opts MODEL.PATCH_EMBED patch_embed_bspm.PatchEmbed3D_BSPM
