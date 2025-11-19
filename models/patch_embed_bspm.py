import torch
import torch.nn as nn

"""
自定义 PatchEmbed，使其支持:
- 单通道输入 in_chans=1
- patch size = (5, 2, 2)
"""

class PatchEmbed3D_BSPM(nn.Module):
    def __init__(self, 
                 img_size=(125, 8, 28),
                 patch_size=(5, 2, 2),
                 in_chans=1,
                 embed_dim=768):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: [B, 1, T, H, W]
        x = self.proj(x)  # → [B, C, T/5, H/2, W/2]
        x = x.flatten(2).transpose(1, 2)  # → [B, N, C]
        return x
