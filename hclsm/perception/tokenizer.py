"""Patchify and embed video frames for the vision encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from hclsm.config import PerceptionConfig


class PatchEmbedding(nn.Module):
    """Patchify video frames and project to embedding dimension.

    Takes (B, T, C, H, W) video input and produces (B, T, N_patches, d_model)
    embeddings where N_patches = (H / patch_size) * (W / patch_size).
    """

    def __init__(self, config: PerceptionConfig, in_channels: int = 3) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.d_model = config.d_model
        self.proj = nn.Conv2d(
            in_channels, config.d_model,
            kernel_size=config.patch_size, stride=config.patch_size,
        )
        self.n_patches_h = config.input_resolution // config.patch_size
        self.n_patches_w = config.input_resolution // config.patch_size

    @property
    def n_patches(self) -> int:
        return self.n_patches_h * self.n_patches_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Patchify and embed.

        Args:
            x: Video frames (B, T, C, H, W).

        Returns:
            Patch embeddings (B, T, N_patches, d_model).
        """
        B, T = x.shape[:2]
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.proj(x)  # (B*T, d_model, H_p, W_p)
        x = rearrange(x, "(b t) d h w -> b t (h w) d", b=B, t=T)
        return x
