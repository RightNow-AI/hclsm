"""Learned value function over world states — stub for Sprint 1."""

from __future__ import annotations

import torch
import torch.nn as nn


class ValueFunction(nn.Module):
    """Predicts scalar value/cost of a world state given a goal.

    Operates on pooled object representations.
    """

    def __init__(self, d_slot: int, d_goal: int = 0) -> None:
        super().__init__()
        d_input = d_slot + d_goal if d_goal > 0 else d_slot
        self.head = nn.Sequential(
            nn.Linear(d_input, d_slot),
            nn.ReLU(),
            nn.Linear(d_slot, d_slot // 4),
            nn.ReLU(),
            nn.Linear(d_slot // 4, 1),
        )

    def forward(
        self,
        obj_states: torch.Tensor,
        alive_mask: torch.Tensor | None = None,
        goal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict state value.

        Args:
            obj_states: (B, N, d_slot).
            alive_mask: (B, N).
            goal: (B, d_goal) optional goal conditioning.

        Returns:
            value: (B, 1) scalar value.
        """
        if alive_mask is not None:
            mask = alive_mask.unsqueeze(-1)
            n = alive_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = (obj_states * mask).sum(dim=1) / n
        else:
            pooled = obj_states.mean(dim=1)

        if goal is not None:
            pooled = torch.cat([pooled, goal], dim=-1)

        return self.head(pooled)
