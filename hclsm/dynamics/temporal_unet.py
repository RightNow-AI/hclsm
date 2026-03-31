"""Multi-scale temporal architecture — minimal stub for Sprint 1."""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalUNet(nn.Module):
    """Placeholder for multi-scale temporal processing.

    Will implement a U-Net-style architecture over the temporal dimension
    for multi-resolution state processing. Deferred to Sprint 3+.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Identity pass-through for Sprint 1."""
        return x
