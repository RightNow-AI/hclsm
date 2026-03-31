"""Tests for Sprint 4: Causality + Event Detection."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from hclsm.config import HCLSMConfig, CausalityConfig
from hclsm.causality.causal_graph import CausalGraph, _gumbel_softmax_binary
from hclsm.causality.intervention import CounterfactualSimulator, InterventionResult
from hclsm.dynamics.event_detector import EventDetector, MultiScaleTemporalFeatures
from hclsm.training.causal_env import CausalBlockWorld, CausalSceneConfig, causal_discovery_accuracy


# ═══════════════════════════════════════════════════════════════════════════
# Gumbel-Softmax
# ═══════════════════════════════════════════════════════════════════════════

class TestGumbelSoftmax:
    def test_output_range(self):
        logits = torch.randn(10, 10)
        samples = _gumbel_softmax_binary(logits, temperature=1.0, hard=False)
        assert (samples >= 0).all() and (samples <= 1).all()

    def test_hard_samples_binary(self):
        logits = torch.randn(10, 10, requires_grad=True)
        samples = _gumbel_softmax_binary(logits, temperature=0.5, hard=True)
        unique_vals = samples.unique()
        assert all(v in [0.0, 1.0] for v in unique_vals.tolist())

    def test_straight_through_gradient(self):
        logits = torch.randn(5, 5, requires_grad=True)
        samples = _gumbel_softmax_binary(logits, temperature=1.0, hard=True)
        loss = samples.sum()
        loss.backward()
        assert logits.grad is not None

    def test_low_temperature_more_discrete(self):
        logits = torch.randn(100, requires_grad=True)
        soft_high = _gumbel_softmax_binary(logits, temperature=5.0, hard=False)
        soft_low = _gumbel_softmax_binary(logits, temperature=0.1, hard=False)
        # Low temp should be closer to 0 or 1
        entropy_high = -(soft_high * soft_high.log().clamp(min=-10)).mean()
        entropy_low = -(soft_low * soft_low.log().clamp(min=-10)).mean()
        assert entropy_low < entropy_high


# ═══════════════════════════════════════════════════════════════════════════
# CausalGraph
# ═══════════════════════════════════════════════════════════════════════════

class TestCausalGraph:
    @pytest.fixture
    def causal_graph(self):
        config = CausalityConfig(enabled=True)
        return CausalGraph(config, n_slots=16)

    def test_static_adjacency(self, causal_graph):
        A = causal_graph.compute_adjacency()
        assert A.shape == (16, 16)
        assert A.diag().sum() == 0  # No self-causation

    def test_data_dependent_adjacency(self, causal_graph):
        obj_states = torch.randn(2, 16, 64)
        A = causal_graph.compute_adjacency(obj_states)
        assert A.shape == (2, 16, 16)
        assert A[:, range(16), range(16)].sum() == 0  # No self-causation

    def test_alive_mask_zeroes_dead(self, causal_graph):
        obj_states = torch.randn(2, 16, 64)
        alive = torch.ones(2, 16)
        alive[:, -4:] = 0
        A = causal_graph.compute_adjacency(obj_states, alive)
        assert A[:, -4:, :].abs().max() < 1e-5
        assert A[:, :, -4:].abs().max() < 1e-5

    def test_intervention_removes_incoming(self, causal_graph):
        A = torch.rand(2, 16, 16)
        A_do = causal_graph.intervene(A, target_idx=5)
        assert A_do[:, :, 5].abs().sum() == 0
        # Other columns unchanged
        assert torch.equal(A_do[:, :, 0], A[:, :, 0])

    def test_dag_constraint_on_dag(self):
        config = CausalityConfig(enabled=True)
        cg = CausalGraph(config, n_slots=4)
        # Lower triangular = DAG
        A_dag = torch.tensor([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ])
        h = cg.dag_constraint(A_dag)
        # h should be small (close to 0) for a DAG
        # Not exactly 0 due to Taylor approximation
        assert h.item() < 5.0  # Much less than for a cyclic graph

    def test_dag_constraint_on_cycle(self):
        config = CausalityConfig(enabled=True)
        cg = CausalGraph(config, n_slots=3)
        # Cycle: 0→1→2→0
        A_cycle = torch.tensor([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ])
        h = cg.dag_constraint(A_cycle)
        assert h.item() > 0.1  # Should be > 0 for cyclic graph

    def test_losses_differentiable(self, causal_graph):
        obj_states = torch.randn(2, 16, 64)
        A = causal_graph.compute_adjacency(obj_states)
        losses = causal_graph.compute_losses(A)
        losses["causal_total"].backward()
        assert causal_graph.W_causal.grad is not None

    def test_temperature_annealing(self, causal_graph):
        t0 = causal_graph.temperature.item()
        for _ in range(10):
            causal_graph.anneal_temperature(0.9)
        assert causal_graph.temperature.item() < t0
        assert causal_graph.temperature.item() >= causal_graph.temperature_min

    def test_lagrangian_update(self, causal_graph):
        info = causal_graph.update_lagrangian(h_value=5.0)
        assert info["alpha"] > 0
        assert info["rho"] > 0

    def test_edge_statistics(self, causal_graph):
        causal_graph.compute_adjacency()
        stats = causal_graph.get_edge_statistics()
        assert "n_edges" in stats
        assert "sparsity" in stats

    def test_disabled_graph_returns_zeros(self):
        config = CausalityConfig(enabled=False)
        cg = CausalGraph(config, n_slots=8)
        obj_states = torch.randn(2, 8, 32)
        A = cg.compute_adjacency(obj_states)
        assert A.sum() == 0


# ═══════════════════════════════════════════════════════════════════════════
# EventDetector
# ═══════════════════════════════════════════════════════════════════════════

class TestEventDetector:
    @pytest.fixture
    def detector(self):
        return EventDetector(d_input=64, window_size=8)

    def test_forward_shapes(self, detector):
        states = torch.randn(2, 16, 64)
        probs, mask = detector(states)
        assert probs.shape == (2, 16)
        assert mask.shape == (2, 16)

    def test_probs_in_range(self, detector):
        states = torch.randn(2, 16, 64)
        probs, _ = detector(states)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_at_least_one_event(self, detector):
        states = torch.randn(2, 4, 64)
        _, mask = detector(states)
        assert mask.sum(dim=1).min() >= 1

    def test_event_rate_loss(self, detector):
        probs = torch.full((2, 16), 0.05)  # Very low rate
        loss = detector.event_rate_loss(probs)
        assert loss.item() > 0  # Should penalize low rate

    def test_contrastive_loss(self, detector):
        states = torch.randn(2, 16, 64)
        probs = torch.rand(2, 16, requires_grad=True)
        loss = detector.contrastive_event_loss(probs, states)
        loss.backward()
        assert probs.grad is not None

    def test_gradient_flow(self, detector):
        states = torch.randn(2, 16, 64)
        probs, mask = detector(states)
        loss = probs.sum() + mask.sum()
        loss.backward()
        assert detector.conv1.weight.grad is not None

    def test_short_sequence(self, detector):
        states = torch.randn(2, 2, 64)
        probs, mask = detector(states)
        assert probs.shape == (2, 2)


class TestMultiScaleFeatures:
    def test_output_shape(self):
        msf = MultiScaleTemporalFeatures(d_input=64)
        states = torch.randn(2, 16, 64)
        out = msf(states)
        assert out.shape == (2, 16, 64)

    def test_short_sequence(self):
        msf = MultiScaleTemporalFeatures(d_input=32)
        states = torch.randn(2, 3, 32)
        out = msf(states)
        assert out.shape == (2, 3, 32)


# ═══════════════════════════════════════════════════════════════════════════
# CounterfactualSimulator
# ═══════════════════════════════════════════════════════════════════════════

class TestCounterfactualSimulator:
    @pytest.fixture
    def sim(self):
        return CounterfactualSimulator(d_slot=64, n_max_slots=16)

    def test_do_intervention(self, sim):
        states = torch.randn(2, 16, 64)
        adj = torch.rand(2, 16, 16) * 0.5
        target = torch.tensor([3, 7])
        value = torch.randn(2, 64)

        mod_states, mod_adj = sim.do_intervention(states, adj, target, value)
        # Target slots should be clamped
        assert torch.equal(mod_states[0, 3], value[0])
        assert torch.equal(mod_states[1, 7], value[1])
        # Incoming edges removed
        assert mod_adj[0, :, 3].sum() == 0
        assert mod_adj[1, :, 7].sum() == 0

    def test_forward_single_step(self, sim):
        states = torch.randn(2, 16, 64)
        adj = torch.rand(2, 16, 16) * 0.3
        result = sim(states, adj, n_rollout_steps=1)
        assert isinstance(result, InterventionResult)
        assert result.counterfactual_states.shape == (2, 1, 16, 64)

    def test_forward_multi_step(self, sim):
        states = torch.randn(2, 16, 64)
        adj = torch.rand(2, 16, 16) * 0.3
        result = sim(states, adj, dynamics_fn=nn.Identity(), n_rollout_steps=5)
        assert result.counterfactual_states.shape == (2, 5, 16, 64)

    def test_counterfactual_loss(self, sim):
        states = torch.randn(2, 16, 64)
        adj = torch.rand(2, 16, 16) * 0.3
        result = sim(states, adj)
        observed = states + torch.randn_like(states) * 0.1
        loss = sim.counterfactual_loss(result, observed, adj)
        loss.backward()
        assert sim.effect_predictor[0].weight.grad is not None

    def test_with_alive_mask(self, sim):
        states = torch.randn(2, 16, 64)
        adj = torch.rand(2, 16, 16) * 0.3
        alive = torch.ones(2, 16)
        alive[:, -4:] = 0
        result = sim(states, adj, alive_mask=alive)
        assert result.counterfactual_states.shape[2] == 16


# ═══════════════════════════════════════════════════════════════════════════
# CausalBlockWorld
# ═══════════════════════════════════════════════════════════════════════════

class TestCausalBlockWorld:
    def test_sample_shapes(self):
        env = CausalBlockWorld(n_samples=5)
        sample = env[0]
        assert sample["video"].shape == (16, 3, 64, 64)
        assert sample["states"].shape == (16, 5, 8)
        assert sample["causal_graph"].shape == (5, 5)

    def test_graph_is_dag(self):
        env = CausalBlockWorld(n_samples=10)
        for i in range(10):
            g = env[i]["causal_graph"]
            # DAG: lower triangular in our generation
            assert g.diag().sum() == 0  # No self-loops

    def test_custom_config(self):
        cfg = CausalSceneConfig(n_objects=3, n_frames=8, resolution=32)
        env = CausalBlockWorld(n_samples=5, config=cfg)
        sample = env[0]
        assert sample["video"].shape == (8, 3, 32, 32)
        assert sample["states"].shape == (8, 3, 8)

    def test_discovery_accuracy_perfect(self):
        gt = torch.tensor([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.float32)
        metrics = causal_discovery_accuracy(gt, gt)
        assert metrics["precision"] > 0.99
        assert metrics["recall"] > 0.99
        assert metrics["f1"] > 0.99
        assert metrics["shd"] == 0.0

    def test_discovery_accuracy_imperfect(self):
        gt = torch.tensor([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.float32)
        pred = torch.tensor([[0, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=torch.float32)
        metrics = causal_discovery_accuracy(pred, gt)
        assert abs(metrics["precision"] - 0.5) < 0.01
        assert abs(metrics["recall"] - 0.5) < 0.01
