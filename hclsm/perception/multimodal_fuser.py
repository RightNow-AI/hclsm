"""Cross-modal projection and fusion to shared d_world space."""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from hclsm.config import HCLSMConfig


class MultimodalFuser(nn.Module):
    """Projects modality embeddings into shared d_world space and fuses.

    Steps:
    1. Linear project each modality's tokens to d_world
    2. Add learned modality-type embedding
    3. Concatenate all tokens along sequence dimension
    4. Apply 2 cross-attention layers for early fusion

    For Sprint 1, only vision modality is active.
    """

    def __init__(self, config: HCLSMConfig) -> None:
        super().__init__()
        self.d_world = config.d_world

        # Per-modality projections
        self.projections = nn.ModuleDict({
            "vision": nn.Linear(config.perception.d_model, config.d_world),
        })

        # Modality-type embeddings
        self.modality_embeddings = nn.ParameterDict({
            "vision": nn.Parameter(torch.randn(1, 1, 1, config.d_world) * 0.02),
        })

        # Fusion cross-attention layers
        self.fusion_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.d_world,
                nhead=max(1, config.d_world // 64),
                dim_feedforward=config.d_world * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(2)
        ])

    def forward(
        self,
        modality_tokens: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Fuse modality tokens into shared space.

        Args:
            modality_tokens: Dict mapping modality name to tensor
                (B, T, N_tokens, d_modality).

        Returns:
            Fused tokens (B, T, N_total_tokens, d_world).
        """
        projected = []
        for name, tokens in modality_tokens.items():
            B, T = tokens.shape[:2]
            proj = self.projections[name](tokens)  # (B, T, N, d_world)
            proj = proj + self.modality_embeddings[name]
            projected.append(proj)

        # Concatenate along token dimension
        fused = torch.cat(projected, dim=2)  # (B, T, N_total, d_world)

        # Apply fusion layers (per-frame)
        B, T, N, D = fused.shape
        fused = rearrange(fused, "b t n d -> (b t) n d")
        for layer in self.fusion_layers:
            fused = layer(fused)
        fused = rearrange(fused, "(b t) n d -> b t n d", b=B, t=T)

        return fused
