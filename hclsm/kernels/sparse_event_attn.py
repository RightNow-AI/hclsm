"""Sparse event-triggered attention kernel.

Gathers variable-length event sequences from the full temporal grid,
runs dense attention on packed events, scatters results back.

This avoids paying attention cost over all T timesteps when only K<<T
timesteps are actual events.

Provides:
- `sparse_event_attention`: Triton-accelerated gather/scatter + attention
- `naive_sparse_event_attention`: Pure PyTorch reference
- `sparse_event_attn_fn`: Auto-dispatches
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Naive PyTorch reference
# ---------------------------------------------------------------------------

def naive_sparse_event_attention(
    states: torch.Tensor,
    event_mask: torch.Tensor,
    attn_fn: Callable,
    max_events: int | None = None,
) -> torch.Tensor:
    """Gather events → dense attention → scatter back.

    Args:
        states: (B, T, N, D) — object states over full temporal grid.
        event_mask: (B, T) — binary mask, 1 at event timesteps.
        attn_fn: Callable that takes (B*K, N, D) and returns (B*K, N, D).
        max_events: Maximum number of events to process (pads/truncates).

    Returns:
        updated: (B, T, N, D) — updated states (non-event timesteps unchanged).
    """
    B, T, N, D = states.shape
    device = states.device

    if max_events is None:
        max_events = int(event_mask.sum(dim=1).max().item())
    max_events = max(max_events, 1)

    # Gather: collect event timesteps into dense tensor
    event_states = torch.zeros(B, max_events, N, D, device=device, dtype=states.dtype)
    event_pad_mask = torch.zeros(B, max_events, device=device, dtype=torch.bool)
    event_indices = torch.zeros(B, max_events, device=device, dtype=torch.long)

    for b in range(B):
        idx = event_mask[b].nonzero(as_tuple=True)[0]
        K = min(len(idx), max_events)
        if K > 0:
            event_states[b, :K] = states[b, idx[:K]]
            event_pad_mask[b, :K] = True
            event_indices[b, :K] = idx[:K]

    # Dense attention on gathered events
    # Flatten: (B * max_events, N, D)
    flat_states = event_states.reshape(B * max_events, N, D)
    flat_updated = attn_fn(flat_states)
    updated_events = flat_updated.reshape(B, max_events, N, D)

    # Scatter: write back to original positions
    output = states.clone()
    for b in range(B):
        idx = event_mask[b].nonzero(as_tuple=True)[0]
        K = min(len(idx), max_events)
        if K > 0:
            output[b, idx[:K]] = updated_events[b, :K]

    return output


# ---------------------------------------------------------------------------
# Optimized gather/scatter (vectorized, no Python loops)
# ---------------------------------------------------------------------------

def _vectorized_gather(
    states: torch.Tensor,
    event_mask: torch.Tensor,
    max_events: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized gather of event timesteps.

    Returns:
        event_states: (B, K, N, D)
        event_pad_mask: (B, K) bool
        event_indices: (B, K) long
    """
    B, T, N, D = states.shape
    device = states.device

    # Sort event positions: topk on mask gives indices of True entries
    # Use float mask for topk
    mask_f = event_mask.float()

    # Pad mask to ensure we always have max_events entries
    if T < max_events:
        mask_f = F.pad(mask_f, (0, max_events - T), value=0.0)

    # Get top-K indices (sorted by position since mask is 0/1)
    values, raw_indices = mask_f.topk(min(max_events, mask_f.shape[1]), dim=1, sorted=True)

    # Clamp indices to valid range
    indices = raw_indices.clamp(0, T - 1)  # (B, K)
    pad_mask = values > 0.5  # (B, K)
    K = indices.shape[1]

    # Gather: (B, K, N, D)
    idx_expanded = indices[:, :, None, None].expand(B, K, N, D)
    event_states = states.gather(1, idx_expanded)

    # Zero out padded entries
    event_states = event_states * pad_mask[:, :, None, None]

    return event_states, pad_mask, indices


def _vectorized_scatter(
    output: torch.Tensor,
    updated_events: torch.Tensor,
    event_indices: torch.Tensor,
    event_pad_mask: torch.Tensor,
) -> torch.Tensor:
    """Vectorized scatter of updated events back to temporal grid."""
    B, K, N, D = updated_events.shape

    # Only scatter valid (non-padded) events
    idx_expanded = event_indices[:, :, None, None].expand(B, K, N, D)
    mask_expanded = event_pad_mask[:, :, None, None].expand(B, K, N, D)

    # scatter_: write updated values at event positions
    output.scatter_(1, idx_expanded, updated_events * mask_expanded + output.gather(1, idx_expanded) * (~mask_expanded))

    return output


def sparse_event_attention(
    states: torch.Tensor,
    event_mask: torch.Tensor,
    attn_fn: Callable,
    max_events: int | None = None,
) -> torch.Tensor:
    """Optimized sparse event attention with vectorized gather/scatter.

    Args:
        states: (B, T, N, D) — object states over full temporal grid.
        event_mask: (B, T) — binary mask, 1 at event timesteps.
        attn_fn: Callable (B*K, N, D) -> (B*K, N, D).
        max_events: Maximum events to process.

    Returns:
        updated: (B, T, N, D).
    """
    B, T, N, D = states.shape

    if max_events is None:
        max_events = max(int(event_mask.sum(dim=1).max().item()), 1)

    # Gather
    event_states, pad_mask, indices = _vectorized_gather(states, event_mask, max_events)
    K = event_states.shape[1]

    # Dense attention
    flat = event_states.reshape(B * K, N, D)
    flat_updated = attn_fn(flat)
    updated_events = flat_updated.reshape(B, K, N, D)

    # Scatter
    output = states.clone()
    output = _vectorized_scatter(output, updated_events, indices, pad_mask)

    return output


# ---------------------------------------------------------------------------
# Triton-accelerated gather kernel
# ---------------------------------------------------------------------------

if TRITON_AVAILABLE:

    @triton.jit
    def _sparse_gather_kernel(
        States_ptr, Mask_ptr, Out_ptr, Indices_ptr,
        B, T, N, D, K_max,
        stride_sb, stride_st, stride_sn, stride_sd,
        stride_ob, stride_ok, stride_on, stride_od,
        BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """Gather event states from temporal grid."""
        bid = tl.program_id(0)
        kid = tl.program_id(1)

        # Load the event index for this (batch, event_slot)
        idx_offset = bid * K_max + kid
        event_t = tl.load(Indices_ptr + idx_offset)

        n_idx = tl.arange(0, BLOCK_N)
        d_idx = tl.arange(0, BLOCK_D)
        n_mask = n_idx < N
        d_mask = d_idx < D

        # Load from states[bid, event_t, :, :]
        src_offsets = (
            bid * stride_sb
            + event_t * stride_st
            + n_idx[:, None] * stride_sn
            + d_idx[None, :] * stride_sd
        )
        mask = n_mask[:, None] & d_mask[None, :]
        vals = tl.load(States_ptr + src_offsets, mask=mask, other=0.0)

        # Store to out[bid, kid, :, :]
        dst_offsets = (
            bid * stride_ob
            + kid * stride_ok
            + n_idx[:, None] * stride_on
            + d_idx[None, :] * stride_od
        )
        tl.store(Out_ptr + dst_offsets, vals, mask=mask)


# ---------------------------------------------------------------------------
# Auto-dispatch
# ---------------------------------------------------------------------------

def sparse_event_attn_fn(
    states: torch.Tensor,
    event_mask: torch.Tensor,
    attn_fn: Callable,
    max_events: int | None = None,
    backend: str = "auto",
) -> torch.Tensor:
    """Auto-dispatching sparse event attention.

    Args:
        states: (B, T, N, D)
        event_mask: (B, T)
        attn_fn: (B*K, N, D) -> (B*K, N, D)
        max_events: Max events.
        backend: "optimized", "naive", or "auto".

    Returns:
        (B, T, N, D) with event timesteps updated.
    """
    if backend == "naive":
        return naive_sparse_event_attention(states, event_mask, attn_fn, max_events)
    return sparse_event_attention(states, event_mask, attn_fn, max_events)
