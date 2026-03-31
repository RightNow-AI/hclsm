"""Action-conditioned dynamics — maps actions to per-object effects."""

from __future__ import annotations

import torch
import torch.nn as nn


class ActionConditioner(nn.Module):
    """Maps actions to per-object state modifications.

    For each object, computes the effect of an action on its state.
    """

    def __init__(self, d_action: int, d_slot: int, n_max_slots: int) -> None:
        super().__init__()
        self.effect_mlp = nn.Sequential(
            nn.Linear(d_action + d_slot, d_slot),
            nn.ReLU(),
            nn.Linear(d_slot, d_slot),
            nn.Tanh(),
        )

    def forward(
        self,
        obj_states: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute action effects on each object.

        Args:
            obj_states: (B, N, d_slot) object states.
            action: (B, d_action) action vector.

        Returns:
            modified: (B, N, d_slot) action-conditioned object states.
        """
        B, N, D = obj_states.shape
        action_expanded = action.unsqueeze(1).expand(B, N, -1)
        combined = torch.cat([obj_states, action_expanded], dim=-1)
        effect = self.effect_mlp(combined)
        return obj_states + effect
