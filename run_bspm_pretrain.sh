#!/bin/bash

python -m torch.distributed.launch \
  --nproc_per_node=8 \
  main_pretrain.py \
  --config configs/pretrain/bspm_vitb_patch522.yaml \
  --opts \
  MODEL.PATCH_EMBED patch_embed_bspm.PatchEmbed3D_BSPM
