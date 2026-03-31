"""Temporal slot tracking via GPU-native Sinkhorn matching.

Sprint 7: Replaced scipy Hungarian (CPU) with differentiable Sinkhorn-Knopp
algorithm that runs entirely on GPU. No CPU transfers, no Python loops over batch.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hclsm.config import ObjectConfig


def _sinkhorn_matching(
    cost: torch.Tensor,
    n_iters: int = 20,
    temperature: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU-native differentiable matching via Sinkhorn-Knopp.

    Args:
        cost: (B, N, N) cost matrix (lower = better match).
        n_iters: Sinkhorn iterations.
        temperature: Lower = harder assignment.

    Returns:
        perm: (B, N) permutation indices (argmax of assignment).
        confidence: (B, N) match confidence.
    """
    # Convert to log-space assignment (negate cost for maximization)
    log_alpha = -cost / max(temperature, 1e-6)

    # Sinkhorn iterations (alternating row/column normalization)
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)

    assignment = log_alpha.exp()  # (B, N, N)
    perm = assignment.argmax(dim=2)  # (B, N)
    confidence = assignment.max(dim=2).values  # (B, N)

    return perm, confidence


class SlotTracker(nn.Module):
    """Match and reorder slots across consecutive frames.

    Uses learned matching score + differentiable Sinkhorn algorithm.
    Runs entirely on GPU — no CPU transfers.
    """

    def __init__(self, config: ObjectConfig) -> None:
        super().__init__()
        self.d_slot = config.d_slot
        self.n_max_slots = config.n_max_slots

        # Matching score MLP: [slot_t; slot_t+1; |diff|] -> scalar
        self.match_mlp = nn.Sequential(
            nn.Linear(config.d_slot * 3, config.d_slot),
            nn.ReLU(),
            nn.Linear(config.d_slot, config.d_slot // 4),
            nn.ReLU(),
            nn.Linear(config.d_slot // 4, 1),
        )

        # GRU for hidden state carry-forward (orthogonal init)
        self.hidden_gru = nn.GRUCell(config.d_slot, config.d_slot)
        nn.init.orthogonal_(self.hidden_gru.weight_ih)
        nn.init.orthogonal_(self.hidden_gru.weight_hh)
        nn.init.zeros_(self.hidden_gru.bias_ih)
        nn.init.zeros_(self.hidden_gru.bias_hh)

    def compute_cost_matrix(
        self,
        slots_t: torch.Tensor,
        slots_t1: torch.Tensor,
        alive_t: torch.Tensor,
        alive_t1: torch.Tensor,
    ) -> torch.Tensor:
        """Compute matching cost matrix between two frames.

        Returns:
            cost: (B, N, N) cost matrix (lower = better match).
        """
        B, N, D = slots_t.shape

        # Clamp inputs for numerical safety
        s_t = torch.clamp(slots_t, -50, 50)
        s_t1 = torch.clamp(slots_t1, -50, 50)

        # Expand for all pairs: (B, N, N, d_slot)
        s_t_exp = s_t[:, :, None, :].expand(B, N, N, D)
        s_t1_exp = s_t1[:, None, :, :].expand(B, N, N, D)
        diff = (s_t_exp - s_t1_exp).abs()

        pair_feat = torch.cat([s_t_exp, s_t1_exp, diff], dim=-1)
        score = self.match_mlp(pair_feat).squeeze(-1)  # (B, N, N)

        # Mask dead slots
        dead_t = (alive_t < 0.5).float()
        dead_t1 = (alive_t1 < 0.5).float()
        penalty = 1e4  # Reduced from 1e6 for numerical stability
        score = score - penalty * dead_t[:, :, None]
        score = score - penalty * dead_t1[:, None, :]

        return -score

    def forward(
        self,
        slots_t: torch.Tensor,
        slots_t1: torch.Tensor,
        alive_t: torch.Tensor,
        alive_t1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Match slots via GPU-native Sinkhorn. No CPU transfer.

        Returns:
            perm: (B, N) permutation indices to reorder slots_t1.
            matched: (B, N) binary mask of matched slots.
        """
        cost = self.compute_cost_matrix(slots_t, slots_t1, alive_t, alive_t1)

        # Clamp cost for Sinkhorn stability
        cost = torch.clamp(cost, -1e4, 1e4)

        perm, confidence = _sinkhorn_matching(cost, n_iters=20, temperature=0.05)

        # Matched = both alive and high confidence
        alive_t_hard = (alive_t > 0.5).float()
        alive_t1_hard = (alive_t1 > 0.5).float()
        matched_t1 = alive_t1_hard.gather(1, perm)
        matched = alive_t_hard * matched_t1 * (confidence > 0.3).float()

        return perm, matched

    def reorder_and_update(
        self,
        slots_t1: torch.Tensor,
        perm: torch.Tensor,
        hidden_t: torch.Tensor,
        matched: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reorder slots and carry forward hidden state."""
        B, N, D = slots_t1.shape

        perm_expanded = perm.unsqueeze(-1).expand(B, N, D)
        reordered = slots_t1.gather(1, perm_expanded)

        hidden_t1 = self.hidden_gru(
            reordered.reshape(-1, D),
            hidden_t.reshape(-1, D),
        ).reshape(B, N, D)

        unmatched = (1 - matched).unsqueeze(-1)
        hidden_t1 = matched.unsqueeze(-1) * hidden_t1 + unmatched * reordered

        return reordered, hidden_t1
