"""Tests for the causality module."""

from __future__ import annotations

import torch
import pytest

from hclsm.config import CausalityConfig
from hclsm.causality.causal_graph import CausalGraph


class TestCausalGraph:
    def test_adjacency_range(self):
        config = CausalityConfig(enabled=True)
        cg = CausalGraph(config, n_slots=16)
        A = cg.adjacency
        assert A.min() >= 0.0
        assert A.max() <= 1.0
        assert A.shape == (16, 16)

    def test_dag_constraint_nonneg(self):
        config = CausalityConfig(enabled=True)
        cg = CausalGraph(config, n_slots=8)
        A = cg.adjacency
        h = cg.dag_constraint(A)
        assert h >= -1e-5  # Should be non-negative (up to numerical error)

    def test_sparsity_loss_nonneg(self):
        config = CausalityConfig(enabled=True)
        cg = CausalGraph(config, n_slots=8)
        losses = cg.compute_losses()
        assert losses["sparsity"] >= 0

    def test_disabled_returns_zeros(self):
        config = CausalityConfig(enabled=False)
        cg = CausalGraph(config, n_slots=8)
        losses = cg.compute_losses()
        assert losses["causal_total"] == 0.0

    def test_gradient_through_losses(self):
        config = CausalityConfig(enabled=True)
        cg = CausalGraph(config, n_slots=8)
        losses = cg.compute_losses()
        losses["causal_total"].backward()
        assert cg.W_causal.grad is not None
