"""Learned causal graph over object slots with DAG constraints.

Sprint 4: Full implementation with data-dependent adjacency, Gumbel-softmax
edge sampling, intervention masks for do-calculus, and augmented Lagrangian
optimization with penalty scheduling.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hclsm.config import CausalityConfig


def _gumbel_softmax_binary(
    logits: torch.Tensor,
    temperature: float = 1.0,
    hard: bool = True,
) -> torch.Tensor:
    """Gumbel-softmax for binary (Bernoulli) sampling.

    Samples from Bernoulli(sigmoid(logits)) with reparameterization.

    Args:
        logits: (*) unnormalized log-probabilities.
        temperature: Softmax temperature (lower = more discrete).
        hard: If True, returns hard samples with straight-through gradient.

    Returns:
        samples: (*) in [0, 1] (soft) or {0, 1} (hard with ST gradient).
    """
    if not logits.requires_grad or not logits.is_floating_point():
        return (logits > 0).float()

    # Sample Gumbel noise for binary case
    u = torch.rand_like(logits).clamp(1e-8, 1 - 1e-8)
    gumbel = -torch.log(-torch.log(u))

    y_soft = torch.sigmoid((logits + gumbel) / temperature)

    if hard:
        y_hard = (y_soft > 0.5).float()
        return y_hard - y_soft.detach() + y_soft  # Straight-through
    return y_soft


class CausalGraph(nn.Module):
    """Learns an explicit causal adjacency matrix over object slots.

    Sprint 4 enhancements:
    - Data-dependent adjacency: edge logits conditioned on object state pairs
    - Gumbel-softmax sampling for differentiable discrete edge structure
    - Temperature annealing (soft → hard during training)
    - Intervention masks for counterfactual simulation (do-calculus)
    - Augmented Lagrangian optimizer for DAG constraint

    The adjacency matrix A_ij represents "object i causally influences object j".
    """

    def __init__(self, config: CausalityConfig, n_slots: int) -> None:
        super().__init__()
        self.enabled = config.enabled
        self.n_slots = n_slots
        self.sparsity_lambda = config.sparsity_lambda

        if self.enabled:
            # Static prior — initialize negative so sigmoid(W) starts sparse (~0.12)
            self.W_causal = nn.Parameter(torch.full((n_slots, n_slots), -2.0))

            # Data-dependent edge logit network
            # Input: [o_i; o_j; o_i - o_j] → scalar logit
            self.edge_net = nn.Sequential(
                nn.Linear(n_slots * 3, n_slots),
                nn.ReLU(),
                nn.Linear(n_slots, 1),
            )
            # Separate MLP that operates on slot features (d_slot dimension)
            self._d_slot_initialized = False
            self._edge_from_slots: nn.Module | None = None

            # Temperature for Gumbel-softmax (annealed during training)
            self.register_buffer(
                "temperature", torch.tensor(config.temperature_init)
            )
            self.temperature_min = config.temperature_min

            # Augmented Lagrangian parameters — gentle schedule
            self.register_buffer(
                "rho", torch.tensor(config.dag_penalty_rho_init)
            )
            self.register_buffer("alpha", torch.tensor(0.0))
            self.rho_max = config.dag_penalty_rho_max
            self.register_buffer("rho_mult", torch.tensor(1.5))  # Gentle: 1.5x not 10x

            # Track DAG constraint for convergence
            self.register_buffer("last_h", torch.tensor(float("inf")))
            self.register_buffer("prev_h", torch.tensor(float("inf")))

        # Cache last computed adjacency for external access
        self.register_buffer(
            "adjacency", torch.zeros(n_slots, n_slots), persistent=False,
        )

    def _lazy_init_edge_net(self, d_slot: int) -> None:
        """Lazily initialize the slot-feature-based edge network."""
        if self._d_slot_initialized:
            return
        self._edge_from_slots = nn.Sequential(
            nn.Linear(d_slot * 3, d_slot),
            nn.ReLU(),
            nn.Linear(d_slot, d_slot // 4),
            nn.ReLU(),
            nn.Linear(d_slot // 4, 1),
        )
        device = self.W_causal.device
        self._edge_from_slots = self._edge_from_slots.to(device)
        self._d_slot_initialized = True

    def compute_adjacency(
        self,
        obj_states: torch.Tensor | None = None,
        alive_mask: torch.Tensor | None = None,
        hard: bool = True,
    ) -> torch.Tensor:
        """Compute the causal adjacency matrix.

        Args:
            obj_states: (B, N, d_slot) object states. If None, uses static prior.
            alive_mask: (B, N) alive mask. Dead slots get zero edges.
            hard: Whether to use hard Gumbel-softmax sampling.

        Returns:
            A: (B, N, N) causal adjacency matrix in [0, 1].
               If obj_states is None, returns (N, N) from static prior.
        """
        if not self.enabled:
            if obj_states is not None:
                B = obj_states.shape[0]
                return torch.zeros(B, self.n_slots, self.n_slots, device=obj_states.device)
            return torch.zeros(self.n_slots, self.n_slots)

        N = self.n_slots

        if obj_states is None:
            # Static prior only
            logits = self.W_causal
            A = _gumbel_softmax_binary(logits, self.temperature.item(), hard)
            # Zero diagonal (no self-causation)
            A = A * (1 - torch.eye(N, device=A.device))
            self.adjacency = A.detach()
            return A

        B, N_actual, d_slot = obj_states.shape
        self._lazy_init_edge_net(d_slot)

        # Clamp slot states to prevent bf16 overflow in edge MLP
        obj_safe = torch.clamp(obj_states, -50, 50)

        # Data-dependent edge logits from object states
        o_i = obj_safe.unsqueeze(2).expand(B, N_actual, N_actual, d_slot)
        o_j = obj_safe.unsqueeze(1).expand(B, N_actual, N_actual, d_slot)
        pair_feat = torch.cat([o_i, o_j, o_i - o_j], dim=-1)  # (B, N, N, 3*d_slot)

        # Data-dependent logits — compute in fp32 for stability
        with torch.amp.autocast("cuda", enabled=False):
            data_logits = self._edge_from_slots(pair_feat.float()).squeeze(-1)  # (B, N, N)

        # Combine with static prior (broadcast W_causal to batch)
        logits = data_logits + self.W_causal[:N_actual, :N_actual].unsqueeze(0)

        # Gumbel-softmax sampling
        temp = self.temperature.item()
        A = _gumbel_softmax_binary(logits, temp, hard)

        # Zero diagonal
        diag_mask = (1 - torch.eye(N_actual, device=A.device)).unsqueeze(0)
        A = A * diag_mask

        # Mask dead slots
        if alive_mask is not None:
            pair_mask = alive_mask.unsqueeze(2) * alive_mask.unsqueeze(1)  # (B, N, N)
            A = A * pair_mask

        # Cache mean adjacency for external access
        self.adjacency = A.detach().mean(dim=0)

        return A

    def intervene(
        self,
        adjacency: torch.Tensor,
        target_idx: int | torch.Tensor,
    ) -> torch.Tensor:
        """Apply do-operator: remove all incoming edges to target node(s).

        Implements do(X_i = x) by zeroing column i of the adjacency matrix
        (removing all causal influences ON node i).

        Args:
            adjacency: (B, N, N) or (N, N) causal adjacency.
            target_idx: Index or indices of intervention targets.

        Returns:
            Modified adjacency with incoming edges removed.
        """
        A = adjacency.clone()
        if isinstance(target_idx, int):
            A[..., :, target_idx] = 0  # Zero column = remove incoming edges
        else:
            for idx in target_idx:
                A[..., :, idx] = 0
        return A

    def dag_constraint(self, A: torch.Tensor) -> torch.Tensor:
        """NOTEARS DAG constraint: h(A) = tr(exp(A * A)) - d.

        Args:
            A: (N, N) or (B, N, N) adjacency matrix.

        Returns:
            h: Scalar constraint value. h=0 iff A is a DAG.
        """
        if A.dim() == 3:
            # Batch: average constraint over batch
            return torch.stack([self.dag_constraint(A[b]) for b in range(A.shape[0])]).mean()

        d = A.shape[0]
        M = A * A  # element-wise square

        # Taylor expansion: exp(M) ~ I + M + M^2/2! + ... + M^k/k!
        # Order 10 for stability
        power = torch.eye(d, device=A.device)
        matrix_exp = torch.eye(d, device=A.device)
        for k in range(1, 11):
            power = power @ M / k
            matrix_exp = matrix_exp + power

        return torch.trace(matrix_exp) - d

    def compute_losses(
        self,
        adjacency: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute causal regularization losses.

        Args:
            adjacency: Pre-computed adjacency. If None, uses cached static adjacency.

        Returns:
            Dict with 'sparsity', 'acyclicity', and 'causal_total' losses.
        """
        if not self.enabled:
            zero = torch.tensor(0.0)
            return {"sparsity": zero, "acyclicity": zero, "causal_total": zero}

        if adjacency is None:
            A = torch.sigmoid(self.W_causal)
        elif adjacency.dim() == 3:
            A = adjacency.mean(dim=0)  # Average over batch for constraint
        else:
            A = adjacency

        device = A.device

        # L1 sparsity
        l_sparsity = self.sparsity_lambda * A.abs().sum()

        # DAG constraint — compute in fp32
        with torch.amp.autocast("cuda", enabled=False):
            A_f = A.float()
            h = self.dag_constraint(A_f)

            if not h.isfinite():
                h = torch.tensor(0.0, device=A.device)

            self.last_h = h.detach()

            # Augmented Lagrangian: alpha * h + rho/2 * h^2
            l_dag = self.alpha.float() * h + (self.rho.float() / 2) * h ** 2

            total = l_sparsity.float() + l_dag
            if not total.isfinite():
                total = l_sparsity.float()

        return {
            "sparsity": l_sparsity,
            "acyclicity": l_dag,
            "dag_h": h.detach(),
            "causal_total": total,
        }

    def update_lagrangian(self, h_value: float | None = None) -> dict[str, float]:
        """Update augmented Lagrangian parameters.

        Call every K steps (not every step). Only escalates rho when h
        hasn't decreased by at least 25% since the last update.

        Args:
            h_value: Current DAG constraint value. If None, uses last_h.

        Returns:
            Dict with updated 'alpha', 'rho', 'h', 'temperature' values.
        """
        if not self.enabled:
            return {}

        if h_value is None:
            h_value = self.last_h.item()

        # Update dual variable (always)
        self.alpha.add_(self.rho * h_value)
        # Clamp alpha to prevent it from growing unboundedly
        self.alpha.clamp_(max=100.0)

        # Only escalate rho if h hasn't improved enough (>75% of previous)
        prev = self.prev_h.item()
        if prev < float("inf") and h_value > 0.75 * prev:
            self.rho.mul_(self.rho_mult)
            self.rho.clamp_(max=self.rho_max)

        self.prev_h = torch.tensor(h_value)

        return {
            "alpha": self.alpha.item(),
            "rho": self.rho.item(),
            "h": h_value,
            "temperature": self.temperature.item(),
        }

    def anneal_temperature(self, factor: float = 0.995) -> None:
        """Anneal Gumbel-softmax temperature towards minimum.

        Call each step or each epoch. Lower temperature → harder edge decisions.
        """
        if not self.enabled:
            return
        self.temperature.mul_(factor)
        self.temperature.clamp_(min=self.temperature_min)

    def get_hard_graph(self, threshold: float = 0.5) -> torch.Tensor:
        """Get a hard binary causal graph for inspection.

        Returns:
            (N, N) binary adjacency matrix.
        """
        return (self.adjacency > threshold).float()

    def get_edge_statistics(self) -> dict[str, float]:
        """Get statistics about the current causal graph."""
        A = self.adjacency
        return {
            "n_edges": (A > 0.5).float().sum().item(),
            "mean_weight": A.mean().item(),
            "max_weight": A.max().item(),
            "sparsity": 1.0 - (A > 0.1).float().mean().item(),
        }
