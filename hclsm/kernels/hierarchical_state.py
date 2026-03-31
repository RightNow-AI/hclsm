"""Hierarchical state management kernel.

Fuses the bottom-up pipeline:
1. Monitor L0 states for event detection
2. Gather event states from temporal grid
3. Compress gathered events via cross-attention to L2 summary tokens
4. Scatter top-down context from L2 back to L0

Provides:
- `hierarchical_state_fused`: Fused pipeline
- `hierarchical_state_naive`: Step-by-step PyTorch reference
- `hierarchical_state_fn`: Auto-dispatch
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable


# ---------------------------------------------------------------------------
# Naive PyTorch reference (step by step)
# ---------------------------------------------------------------------------

def hierarchical_state_naive(
    level0_states: torch.Tensor,
    event_scores: torch.Tensor,
    event_threshold: float,
    gather_fn: Callable,
    level1_fn: Callable,
    level2_fn: Callable,
    scatter_fn: Callable,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Step-by-step hierarchical state management.

    Args:
        level0_states: (B, T, N, D) — L0 SSM output.
        event_scores: (B, T) — event detection scores.
        event_threshold: Threshold for event detection.
        gather_fn: (states, mask) -> (event_states, pad_mask, K_max)
        level1_fn: (event_states, pad_mask) -> updated_events
        level2_fn: (event_states, pad_mask) -> abstract_states
        scatter_fn: (level0, level1_out, level2_out, event_mask) -> combined

    Returns:
        final_states: (B, T, N, D) combined output.
        event_mask: (B, T) detected events.
        level1_out: (B, K, N, D) event-level updates.
        level2_out: (B, n_summary, d_l2) abstract states.
    """
    # Step 1: Event detection
    event_mask = (event_scores > event_threshold).float()

    # Step 2: Gather event states
    event_states, event_pad_mask, K_max = gather_fn(level0_states, event_mask)

    # Step 3: Level 1 processing
    level1_out = level1_fn(event_states, event_pad_mask)

    # Step 4: Level 2 compression
    level2_out = level2_fn(event_states, event_pad_mask)

    # Step 5: Scatter and combine
    final_states = scatter_fn(level0_states, level1_out, level2_out, event_mask)

    return final_states, event_mask, level1_out, level2_out


# ---------------------------------------------------------------------------
# Fused pipeline (overlaps compute where possible)
# ---------------------------------------------------------------------------

def hierarchical_state_fused(
    level0_states: torch.Tensor,
    event_scores: torch.Tensor,
    event_threshold: float,
    gather_fn: Callable,
    level1_fn: Callable,
    level2_fn: Callable,
    scatter_fn: Callable,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused hierarchical state management.

    Combines event detection, gathering, and processing into a tighter
    pipeline that avoids redundant memory reads/writes.

    Same interface as hierarchical_state_naive.
    """
    B, T, N, D = level0_states.shape
    device = level0_states.device

    # ── Fused event detection + gather ──
    # Avoid materializing the full event_mask, directly compute indices
    event_mask = (event_scores > event_threshold).float()

    # Compute per-batch event counts
    n_events = event_mask.sum(dim=1).long()  # (B,)
    K_max = max(n_events.max().item(), 1)

    # Vectorized gather using topk
    scores_for_gather = event_scores * event_mask + (1 - event_mask) * (-1e9)
    if T < K_max:
        scores_for_gather = F.pad(scores_for_gather, (0, K_max - T), value=-1e9)

    _, event_indices = scores_for_gather.topk(K_max, dim=1, sorted=True)
    event_indices = event_indices.clamp(0, T - 1)

    # Gather states at event indices
    idx_expanded = event_indices[:, :, None, None].expand(B, K_max, N, D)
    event_states = level0_states.gather(1, idx_expanded)

    # Pad mask
    event_pad_mask = torch.arange(K_max, device=device).unsqueeze(0) < n_events.unsqueeze(1)
    event_states = event_states * event_pad_mask[:, :, None, None]

    # ── Level 1 + Level 2 (can overlap on separate CUDA streams) ──
    level1_out = level1_fn(event_states, event_pad_mask)
    level2_out = level2_fn(event_states, event_pad_mask)

    # ── Scatter and combine ──
    final_states = scatter_fn(level0_states, level1_out, level2_out, event_mask)

    return final_states, event_mask, level1_out, level2_out


# ---------------------------------------------------------------------------
# Stream-overlapped version (Level 1 and Level 2 in parallel)
# ---------------------------------------------------------------------------

def hierarchical_state_streamed(
    level0_states: torch.Tensor,
    event_scores: torch.Tensor,
    event_threshold: float,
    gather_fn: Callable,
    level1_fn: Callable,
    level2_fn: Callable,
    scatter_fn: Callable,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hierarchical state with CUDA stream overlap for Level 1 and Level 2.

    Uses separate CUDA streams for Level 1 and Level 2 processing since
    they are independent given the gathered event states.
    """
    B, T, N, D = level0_states.shape
    device = level0_states.device

    # Event detection + gather (same as fused)
    event_mask = (event_scores > event_threshold).float()
    n_events = event_mask.sum(dim=1).long()
    K_max = max(n_events.max().item(), 1)

    scores_for_gather = event_scores * event_mask + (1 - event_mask) * (-1e9)
    if T < K_max:
        scores_for_gather = F.pad(scores_for_gather, (0, K_max - T), value=-1e9)

    _, event_indices = scores_for_gather.topk(K_max, dim=1, sorted=True)
    event_indices = event_indices.clamp(0, T - 1)

    idx_expanded = event_indices[:, :, None, None].expand(B, K_max, N, D)
    event_states = level0_states.gather(1, idx_expanded)
    event_pad_mask = torch.arange(K_max, device=device).unsqueeze(0) < n_events.unsqueeze(1)
    event_states = event_states * event_pad_mask[:, :, None, None]

    # Overlap Level 1 and Level 2 if on CUDA
    if device.type == "cuda":
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()

        with torch.cuda.stream(s1):
            level1_out = level1_fn(event_states, event_pad_mask)

        with torch.cuda.stream(s2):
            level2_out = level2_fn(event_states, event_pad_mask)

        torch.cuda.current_stream().wait_stream(s1)
        torch.cuda.current_stream().wait_stream(s2)
    else:
        level1_out = level1_fn(event_states, event_pad_mask)
        level2_out = level2_fn(event_states, event_pad_mask)

    final_states = scatter_fn(level0_states, level1_out, level2_out, event_mask)
    return final_states, event_mask, level1_out, level2_out


# ---------------------------------------------------------------------------
# Auto-dispatch
# ---------------------------------------------------------------------------

def hierarchical_state_fn(
    level0_states: torch.Tensor,
    event_scores: torch.Tensor,
    event_threshold: float,
    gather_fn: Callable,
    level1_fn: Callable,
    level2_fn: Callable,
    scatter_fn: Callable,
    backend: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Auto-dispatching hierarchical state management.

    Args:
        backend: "streamed", "fused", "naive", or "auto".
    """
    if backend == "naive":
        return hierarchical_state_naive(
            level0_states, event_scores, event_threshold,
            gather_fn, level1_fn, level2_fn, scatter_fn,
        )

    if backend == "streamed" or (backend == "auto" and level0_states.is_cuda):
        return hierarchical_state_streamed(
            level0_states, event_scores, event_threshold,
            gather_fn, level1_fn, level2_fn, scatter_fn,
        )

    return hierarchical_state_fused(
        level0_states, event_scores, event_threshold,
        gather_fn, level1_fn, level2_fn, scatter_fn,
    )
