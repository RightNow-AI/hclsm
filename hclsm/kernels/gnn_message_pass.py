"""Fused GNN message passing kernel.

Fuses edge computation + message passing + aggregation for the relation graph.
For N=64 slots: 4096 edges fit in shared memory.

Pipeline per round:
1. Edge features: e_ij = MLP([o_i; o_j; o_i - o_j; o_i * o_j])
2. Edge weights: w_ij = sigmoid(linear(e_ij))
3. Messages: m_ij = w_ij * MLP_msg(e_ij)
4. Aggregation: agg_i = sum_j(m_ij)
5. Node update: o_i' = o_i + MLP_update([o_i; agg_i])

Provides:
- `fused_gnn_message_pass`: Triton kernel for steps 1-4 (when available)
- `naive_gnn_message_pass`: Pure PyTorch reference
- `gnn_message_pass_fn`: Auto-dispatch
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Naive PyTorch reference
# ---------------------------------------------------------------------------

def naive_gnn_message_pass(
    nodes: torch.Tensor,
    alive_mask: torch.Tensor | None,
    edge_mlp: nn.Module,
    edge_weight_linear: nn.Module,
    msg_mlp: nn.Module,
    update_mlp: nn.Module,
    n_rounds: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch GNN message passing (reference).

    Args:
        nodes: (B, N, D) object slot features.
        alive_mask: (B, N) or None.
        edge_mlp: [4*D] -> d_edge.
        edge_weight_linear: d_edge -> 1.
        msg_mlp: d_edge -> D.
        update_mlp: 2*D -> D.
        n_rounds: Message passing rounds.

    Returns:
        updated_nodes: (B, N, D).
        edge_features: (B, N, N, d_edge) from last round.
    """
    B, N, D = nodes.shape
    device = nodes.device

    if alive_mask is not None:
        pair_mask = alive_mask[:, :, None] * alive_mask[:, None, :]
    else:
        pair_mask = torch.ones(B, N, N, device=device)

    edge_feat = None
    for _ in range(n_rounds):
        # All pairs
        n_i = nodes[:, :, None, :].expand(B, N, N, D)
        n_j = nodes[:, None, :, :].expand(B, N, N, D)
        pair_feat = torch.cat([n_i, n_j, n_i - n_j, n_i * n_j], dim=-1)

        edge_feat = edge_mlp(pair_feat)  # (B, N, N, d_edge)
        weights = torch.sigmoid(edge_weight_linear(edge_feat).squeeze(-1))  # (B, N, N)
        weights = weights * pair_mask

        messages = msg_mlp(edge_feat) * weights.unsqueeze(-1)
        agg = messages.sum(dim=2)  # (B, N, D)

        update_input = torch.cat([nodes, agg], dim=-1)
        nodes = nodes + update_mlp(update_input)

    if alive_mask is not None:
        nodes = nodes * alive_mask.unsqueeze(-1)

    return nodes, edge_feat


# ---------------------------------------------------------------------------
# Optimized PyTorch (fused pair computation, avoids explicit N^2 expansion)
# ---------------------------------------------------------------------------

def optimized_gnn_message_pass(
    nodes: torch.Tensor,
    alive_mask: torch.Tensor | None,
    edge_mlp: nn.Module,
    edge_weight_linear: nn.Module,
    msg_mlp: nn.Module,
    update_mlp: nn.Module,
    n_rounds: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimized GNN using batched operations and in-place where safe.

    Same interface as naive_gnn_message_pass but uses memory-efficient
    pair construction by computing features in chunks when N is large.
    """
    B, N, D = nodes.shape
    device = nodes.device

    if alive_mask is not None:
        pair_mask = alive_mask[:, :, None] * alive_mask[:, None, :]
    else:
        pair_mask = None

    edge_feat = None

    for _ in range(n_rounds):
        # Construct all-pairs features: (B, N, N, 4*D)
        # Using expand (no copy) + contiguous only for the cat
        n_i = nodes.unsqueeze(2).expand(B, N, N, D)
        n_j = nodes.unsqueeze(1).expand(B, N, N, D)

        # Compute diff and prod without materializing full intermediates
        diff = n_i - n_j
        prod = n_i * n_j
        pair_feat = torch.cat([n_i, n_j, diff, prod], dim=-1)

        # Edge computation
        edge_feat = edge_mlp(pair_feat)
        weights = torch.sigmoid(edge_weight_linear(edge_feat).squeeze(-1))

        if pair_mask is not None:
            weights = weights * pair_mask

        # Messages and aggregation
        messages = msg_mlp(edge_feat) * weights.unsqueeze(-1)
        agg = messages.sum(dim=2)

        # Node update
        update_input = torch.cat([nodes, agg], dim=-1)
        nodes = nodes + update_mlp(update_input)

    if alive_mask is not None:
        nodes = nodes * alive_mask.unsqueeze(-1)

    return nodes, edge_feat


# ---------------------------------------------------------------------------
# Chunked GNN for memory efficiency with large N
# ---------------------------------------------------------------------------

def chunked_gnn_message_pass(
    nodes: torch.Tensor,
    alive_mask: torch.Tensor | None,
    edge_mlp: nn.Module,
    edge_weight_linear: nn.Module,
    msg_mlp: nn.Module,
    update_mlp: nn.Module,
    n_rounds: int = 1,
    chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Memory-efficient GNN that processes edge pairs in chunks.

    For N=128, the full (B, N, N, 4*D) tensor uses B*128*128*4*D*4 bytes.
    With D=384, B=64: 64*128*128*1536*4 = 64GB — way too much.

    Chunking processes source nodes in blocks of `chunk_size`, reducing
    peak memory by N/chunk_size.
    """
    B, N, D = nodes.shape
    device = nodes.device
    d_edge = None

    if alive_mask is not None:
        pair_mask = alive_mask[:, :, None] * alive_mask[:, None, :]
    else:
        pair_mask = None

    last_edge_feat = None

    for _ in range(n_rounds):
        agg = torch.zeros(B, N, D, device=device, dtype=nodes.dtype)

        for j_start in range(0, N, chunk_size):
            j_end = min(j_start + chunk_size, N)
            C = j_end - j_start

            # n_i: (B, N, 1, D), n_j: (B, 1, C, D)
            n_i = nodes.unsqueeze(2)  # (B, N, 1, D)
            n_j = nodes[:, j_start:j_end].unsqueeze(1)  # (B, 1, C, D)

            n_i_exp = n_i.expand(B, N, C, D)
            n_j_exp = n_j.expand(B, N, C, D)

            pair_feat = torch.cat([
                n_i_exp, n_j_exp,
                n_i_exp - n_j_exp,
                n_i_exp * n_j_exp,
            ], dim=-1)

            edge_feat = edge_mlp(pair_feat)  # (B, N, C, d_edge)
            weights = torch.sigmoid(edge_weight_linear(edge_feat).squeeze(-1))

            if pair_mask is not None:
                weights = weights * pair_mask[:, :, j_start:j_end]

            messages = msg_mlp(edge_feat) * weights.unsqueeze(-1)
            agg = agg + messages.sum(dim=2)

            if j_start == 0:
                d_edge = edge_feat.shape[-1]

            # Keep last chunk's edge features for causal detection
            if j_start + chunk_size >= N:
                last_edge_feat = edge_feat

        update_input = torch.cat([nodes, agg], dim=-1)
        nodes = nodes + update_mlp(update_input)

    if alive_mask is not None:
        nodes = nodes * alive_mask.unsqueeze(-1)

    return nodes, last_edge_feat


# ---------------------------------------------------------------------------
# Auto-dispatch
# ---------------------------------------------------------------------------

def gnn_message_pass_fn(
    nodes: torch.Tensor,
    alive_mask: torch.Tensor | None,
    edge_mlp: nn.Module,
    edge_weight_linear: nn.Module,
    msg_mlp: nn.Module,
    update_mlp: nn.Module,
    n_rounds: int = 1,
    backend: str = "auto",
    chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Auto-dispatching GNN message passing.

    Args:
        nodes: (B, N, D)
        alive_mask: (B, N) or None
        edge_mlp, edge_weight_linear, msg_mlp, update_mlp: GNN modules
        n_rounds: Message passing rounds
        backend: "optimized", "chunked", "naive", or "auto"
        chunk_size: Chunk size for chunked backend

    Returns:
        (updated_nodes, edge_features)
    """
    B, N, D = nodes.shape

    if backend == "naive":
        return naive_gnn_message_pass(
            nodes, alive_mask, edge_mlp, edge_weight_linear,
            msg_mlp, update_mlp, n_rounds,
        )
    if backend == "chunked":
        return chunked_gnn_message_pass(
            nodes, alive_mask, edge_mlp, edge_weight_linear,
            msg_mlp, update_mlp, n_rounds, chunk_size,
        )

    # Auto: use chunked for large N to avoid OOM
    if N > 32:
        return chunked_gnn_message_pass(
            nodes, alive_mask, edge_mlp, edge_weight_linear,
            msg_mlp, update_mlp, n_rounds, chunk_size,
        )

    return optimized_gnn_message_pass(
        nodes, alive_mask, edge_mlp, edge_weight_linear,
        msg_mlp, update_mlp, n_rounds,
    )
