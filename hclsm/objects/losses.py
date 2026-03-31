"""Object-level loss functions: diversity and tracking consistency."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def slot_diversity_loss(
    slots: torch.Tensor, alive: torch.Tensor, margin: float = 0.1,
) -> torch.Tensor:
    """Push slot representations apart to encourage diversity.

    L_diversity = -sum_{i!=j} max(0, cos(z_i, z_j) - margin)

    Args:
        slots: (B, N, d_slot) slot representations.
        alive: (B, N) alive mask.
        margin: Cosine similarity margin.

    Returns:
        Scalar loss.
    """
    B, N, D = slots.shape

    # Normalize for cosine similarity
    slots_norm = F.normalize(slots, dim=-1)  # (B, N, D)

    # Pairwise cosine similarity
    cos_sim = torch.bmm(slots_norm, slots_norm.transpose(1, 2))  # (B, N, N)

    # Mask: only consider alive-alive pairs, exclude diagonal
    alive_hard = (alive > 0.5).float()
    pair_mask = alive_hard[:, :, None] * alive_hard[:, None, :]
    diag_mask = 1.0 - torch.eye(N, device=slots.device).unsqueeze(0)
    mask = pair_mask * diag_mask

    # Hinge loss
    loss = torch.clamp(cos_sim - margin, min=0.0) * mask

    # Average over valid pairs
    n_pairs = mask.sum().clamp(min=1.0)
    return loss.sum() / n_pairs


def slot_tracking_loss(
    slots: torch.Tensor,
    alive: torch.Tensor,
) -> torch.Tensor:
    """Temporal consistency loss between consecutive frames.

    L_tracking = mean_t mean_i ||o_i^t - o_i^{t+1}||^2

    Assumes slots have already been matched/reordered by SlotTracker.

    Args:
        slots: (B, T, N, d_slot) tracked slot sequence.
        alive: (B, T, N) alive mask.

    Returns:
        Scalar loss.
    """
    if slots.shape[1] < 2:
        return torch.tensor(0.0, device=slots.device)

    # Consecutive frame differences — use MEAN over d_slot, not SUM
    # Clamp slots to prevent NaN from bf16 overflow
    slots_safe = torch.clamp(slots, -100, 100)
    diff = slots_safe[:, 1:] - slots_safe[:, :-1]  # (B, T-1, N, d_slot)
    sq_dist = (diff * diff).mean(dim=-1)  # (B, T-1, N) — stable squaring

    # Mask: both frames must be alive
    alive_both = (alive[:, 1:] > 0.5).float() * (alive[:, :-1] > 0.5).float()

    n_valid = alive_both.sum().clamp(min=1.0)
    loss = (sq_dist * alive_both).sum() / n_valid

    # Guard against NaN and clamp magnitude
    if not loss.isfinite():
        return torch.tensor(0.0, device=slots.device, requires_grad=True)
    return torch.clamp(loss, max=10.0)
