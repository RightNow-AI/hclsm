"""Robot joint/sensor encoder — placeholder for future implementation."""

from __future__ import annotations

import torch
import torch.nn as nn

from hclsm.config import PerceptionConfig


class ProprioceptionEncoder(nn.Module):
    """Placeholder proprioception encoder.

    Will encode robot joint states and sensor data into the shared latent space.
    Not implemented in Sprint 1.
    """

    def __init__(self, config: PerceptionConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "ProprioceptionEncoder is not implemented in Sprint 1"
        )
