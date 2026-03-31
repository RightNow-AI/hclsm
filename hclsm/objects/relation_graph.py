"""Learned relation graph (GNN) over object slots."""

from __future__ import annotations

import torch
import torch.nn as nn

from hclsm.config import ObjectConfig


class RelationGraph(nn.Module):
    """Models pairwise object interactions as a learned graph.

    Architecture:
    - Edge computation: e_ij = MLP([o_i; o_j; o_i - o_j; o_i * o_j])
    - Edge weight: w_ij = sigmoid(linear(e_ij))
    - Message passing: m_ij = w_ij * MLP_msg(e_ij), agg_i = sum_j(m_ij)
    - Node update: o_i' = o_i + MLP_update([o_i; agg_i])
    - Causal edge: causal_ij = sigmoid(MLP_causal(e_ij))
    """

    def __init__(self, config: ObjectConfig) -> None:
        super().__init__()
        self.d_slot = config.d_slot
        self.d_edge = config.d_edge
        self.n_rounds = config.gnn_rounds
        self.n_max_slots = config.n_max_slots

        # Pair feature normalization to prevent n_i * n_j explosion
        self.pair_norm = nn.LayerNorm(config.d_slot * 4)

        # Edge feature computation: [o_i; o_j; o_i - o_j; o_i * o_j] -> d_edge
        self.edge_mlp = nn.Sequential(
            nn.Linear(config.d_slot * 4, config.d_edge * 2),
            nn.ReLU(),
            nn.Linear(config.d_edge * 2, config.d_edge),
            nn.ReLU(),
        )

        # Edge weight
        self.edge_weight = nn.Linear(config.d_edge, 1)

        # Message MLP (per round, shared across rounds for simplicity)
        self.msg_mlp = nn.Sequential(
            nn.Linear(config.d_edge, config.d_slot),
            nn.ReLU(),
        )

        # Node update MLP
        self.update_mlp = nn.Sequential(
            nn.Linear(config.d_slot * 2, config.d_slot),
            nn.ReLU(),
            nn.Linear(config.d_slot, config.d_slot),
        )

        # Causal edge detection
        self.causal_mlp = nn.Sequential(
            nn.Linear(config.d_edge, config.d_edge // 2),
            nn.ReLU(),
            nn.Linear(config.d_edge // 2, 1),
        )

    def _compute_edges(
        self, nodes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute edge features and weights for all pairs.

        Args:
            nodes: (B, N, d_slot).

        Returns:
            edge_features: (B, N, N, d_edge).
            edge_weights: (B, N, N).
        """
        B, N, D = nodes.shape

        # All pairs
        n_i = nodes[:, :, None, :].expand(B, N, N, D)  # (B, N, N, d_slot)
        n_j = nodes[:, None, :, :].expand(B, N, N, D)

        pair_feat = torch.cat([n_i, n_j, n_i - n_j, n_i * n_j], dim=-1)
        pair_feat = self.pair_norm(pair_feat)
        edge_feat = self.edge_mlp(pair_feat)  # (B, N, N, d_edge)
        weights = torch.sigmoid(self.edge_weight(edge_feat).squeeze(-1))  # (B, N, N)

        return edge_feat, weights

    def _chunked_message_pass(
        self,
        nodes: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int = 16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Memory-efficient GNN via chunked edge computation.

        Processes source nodes in blocks to avoid materializing the full
        (B, N, N, 4*D) pair tensor. For N=64, D=256 this reduces peak
        memory from ~4GB to ~250MB per chunk.
        """
        B, N, D = nodes.shape
        device = nodes.device
        agg = torch.zeros(B, N, D, device=device, dtype=nodes.dtype)
        last_edge_feat = None

        for j_start in range(0, N, chunk_size):
            j_end = min(j_start + chunk_size, N)
            C = j_end - j_start

            n_i = nodes.unsqueeze(2).expand(B, N, C, D)
            n_j = nodes[:, j_start:j_end].unsqueeze(1).expand(B, N, C, D)

            pair_feat = torch.cat([n_i, n_j, n_i - n_j, n_i * n_j], dim=-1)
            pair_feat = self.pair_norm(pair_feat)
            edge_feat = self.edge_mlp(pair_feat)
            weights = torch.sigmoid(self.edge_weight(edge_feat).squeeze(-1))
            weights = weights * pair_mask[:, :, j_start:j_end]

            messages = self.msg_mlp(edge_feat) * weights.unsqueeze(-1)
            agg = agg + messages.sum(dim=2)
            last_edge_feat = edge_feat

        return agg, last_edge_feat

    def forward(
        self,
        nodes: torch.Tensor,
        alive_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run GNN message passing.

        Uses chunked computation for N > 32 to avoid OOM.

        Args:
            nodes: (B, N, d_slot) object slot features.
            alive_mask: (B, N) alive mask. If None, all slots active.

        Returns:
            updated_nodes: (B, N, d_slot).
            causal_edges: (B, N, N) causal edge probabilities.
        """
        B, N, D = nodes.shape
        use_chunked = N > 32

        # Mask for valid interactions
        if alive_mask is not None:
            pair_mask = alive_mask[:, :, None] * alive_mask[:, None, :]
        else:
            pair_mask = torch.ones(B, N, N, device=nodes.device)

        if use_chunked:
            # Chunked path — memory-efficient for large N
            for _ in range(self.n_rounds):
                agg, _ = self._chunked_message_pass(nodes, pair_mask)
                update_input = torch.cat([nodes, agg], dim=-1)
                nodes = nodes + self.update_mlp(update_input)

            # Causal edges: compute in chunks too
            causal_edges = torch.zeros(B, N, N, device=nodes.device)
            for j_start in range(0, N, 16):
                j_end = min(j_start + 16, N)
                C = j_end - j_start
                n_i = nodes.unsqueeze(2).expand(B, N, C, D)
                n_j = nodes[:, j_start:j_end].unsqueeze(1).expand(B, N, C, D)
                pair_feat = torch.cat([n_i, n_j, n_i - n_j, n_i * n_j], dim=-1)
                pair_feat = self.pair_norm(pair_feat)
                edge_feat = self.edge_mlp(pair_feat)
                causal_edges[:, :, j_start:j_end] = torch.sigmoid(
                    self.causal_mlp(edge_feat).squeeze(-1)
                )
            causal_edges = causal_edges * pair_mask
        else:
            # Original path — fast for small N
            edge_feat, edge_weights = self._compute_edges(nodes)
            edge_weights = edge_weights * pair_mask

            for _ in range(self.n_rounds):
                messages = self.msg_mlp(edge_feat) * edge_weights.unsqueeze(-1)
                agg = messages.sum(dim=2)
                update_input = torch.cat([nodes, agg], dim=-1)
                nodes = nodes + self.update_mlp(update_input)
                if self.n_rounds > 1:
                    edge_feat, edge_weights = self._compute_edges(nodes)
                    edge_weights = edge_weights * pair_mask

            causal_edges = torch.sigmoid(
                self.causal_mlp(edge_feat).squeeze(-1)
            )
            causal_edges = causal_edges * pair_mask

        if alive_mask is not None:
            nodes = nodes * alive_mask.unsqueeze(-1)

        return nodes, causal_edges
