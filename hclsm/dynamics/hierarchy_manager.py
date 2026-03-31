"""Hierarchy manager — orchestrates cross-level information flow.

Handles top-down gating (goal -> event -> physics) and event
gathering/scattering between temporal grids.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


class HierarchyManager(nn.Module):
    """Manages bidirectional information flow between dynamics levels.

    Top-down: Level 2 context -> condition Level 1 -> condition Level 0.
    Bottom-up: Level 0 states -> event detection -> Level 1 -> compression -> Level 2.

    Gating prevents hierarchy from collapsing to Level 0 only.
    """

    def __init__(self, d_slot: int, d_l2: int) -> None:
        super().__init__()
        self.d_slot = d_slot

        # L2 -> L1 top-down projection
        self.l2_to_l1 = nn.Linear(d_l2, d_slot, bias=False)

        # L1 -> L0 gate
        self.gate_l0 = nn.Sequential(
            nn.Linear(d_slot * 2, d_slot),
            nn.Sigmoid(),
        )

        # L2 -> L0 gate (when L2 is active)
        self.gate_l2_l0 = nn.Sequential(
            nn.Linear(d_slot * 2, d_slot),
            nn.Sigmoid(),
        )

    def gather_events(
        self,
        states: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Gather states at event timesteps into a dense tensor.

        Args:
            states: (B, T, N, d_slot) full temporal grid.
            event_mask: (B, T) binary event mask.

        Returns:
            event_states: (B, K_max, N, d_slot) gathered states.
            event_pad_mask: (B, K_max) True for real events.
            K_max: maximum number of events across batch.
        """
        B, T, N, D = states.shape

        # Count events per sample
        event_bool = event_mask > 0.5  # (B, T)
        n_events = event_bool.sum(dim=1)  # (B,)
        K_max = max(int(n_events.max().item()), 1)

        # Vectorized gather: sort mask descending so True comes first
        sorted_mask, sort_indices = event_bool.float().sort(dim=1, descending=True)
        gather_indices = sort_indices[:, :K_max].clamp(0, T - 1)  # (B, K_max)

        # Expand for (B, K_max, N, D) gather
        idx_exp = gather_indices[:, :, None, None].expand(B, K_max, N, D)
        event_states = states.gather(1, idx_exp)

        # Pad mask
        event_pad_mask = sorted_mask[:, :K_max]
        event_states = event_states * event_pad_mask[:, :, None, None]

        return event_states, event_pad_mask, K_max

    def scatter_events(
        self,
        base_states: torch.Tensor,
        event_updates: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter event-level updates back to full temporal grid.

        Level 1 output OVERRIDES Level 0 at event timesteps.

        Args:
            base_states: (B, T, N, d_slot) Level 0 predictions.
            event_updates: (B, K, N, d_slot) Level 1 outputs.
            event_mask: (B, T) binary event mask.

        Returns:
            combined: (B, T, N, d_slot) with event updates scattered in.
        """
        combined = base_states.clone()
        B, T = event_mask.shape[:2]
        N, D = base_states.shape[2], base_states.shape[3]
        K = event_updates.shape[1]

        # Vectorized scatter: get sorted event indices (same order as gather)
        event_bool = event_mask > 0.5
        _, sort_indices = event_bool.float().sort(dim=1, descending=True)
        scatter_indices = sort_indices[:, :K].clamp(0, T - 1)  # (B, K)

        # Expand for (B, K, N, D) scatter
        idx_exp = scatter_indices[:, :, None, None].expand(B, K, N, D)
        pad_mask = (torch.arange(K, device=event_mask.device).unsqueeze(0)
                    < event_bool.sum(dim=1, keepdim=True))  # (B, K)
        mask_exp = pad_mask[:, :, None, None].expand(B, K, N, D)

        # Out-of-place scatter to avoid breaking autograd
        source = torch.where(mask_exp, event_updates, combined.gather(1, idx_exp))
        result = combined.clone()
        result.scatter_(1, idx_exp, source)

        return result

    def forward(
        self,
        level0_states: torch.Tensor,
        level1_states: torch.Tensor,
        level2_context: torch.Tensor | None,
        event_mask: torch.Tensor,
        obj_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply hierarchical gating to produce final predictions.

        Args:
            level0_states: (B, T, N, d_slot) SSM predictions.
            level1_states: (B, K, N, d_slot) event Transformer outputs.
            level2_context: (B, n_summary, d_l2) or None.
            event_mask: (B, T) binary event mask.
            obj_mask: (B, N) alive mask.

        Returns:
            final_states: (B, T, N, d_slot) gated combination.
        """
        B, T, N, D = level0_states.shape

        # Start from L0 with L1 overrides at event times
        combined = self.scatter_events(level0_states, level1_states, event_mask)

        # Apply L2 top-down context if available
        if level2_context is not None:
            # Pool L2 context: (B, d_l2) -> project to (B, d_slot)
            l2_pooled = level2_context.mean(dim=1)  # (B, d_l2)
            l2_proj = self.l2_to_l1(l2_pooled)  # (B, d_slot)

            # Broadcast to (B, T, N, d_slot)
            l2_broadcast = l2_proj[:, None, None, :].expand_as(combined)

            gate_input = torch.cat([combined, l2_broadcast], dim=-1)
            gate = self.gate_l2_l0(gate_input)
            combined = gate * l2_broadcast + (1 - gate) * combined

        # Mask dead objects
        if obj_mask is not None:
            combined = combined * obj_mask[:, None, :, None]

        return combined
