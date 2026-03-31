"""Tests for Sprint 6: Continual Learning + Benchmarks."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from hclsm.memory.episodic import EpisodicMemory
from hclsm.memory.replay_buffer import ReplayBuffer, Experience
from hclsm.memory.semantic import SemanticConsolidation, ConsolidationLoop, EWCRegularizer
from hclsm.training.benchmarks import (
    BenchmarkRunner,
    BenchmarkResult,
    PhysicsPredictionBenchmark,
    CausalDiscoveryBenchmark,
    PlanningBenchmark,
    ContinualLearningBenchmark,
)


class TestEpisodicMemory:
    def test_write_read(self):
        mem = EpisodicMemory(capacity=100, d_memory=64)
        patterns = torch.randn(5, 64)
        mem.write(patterns)
        assert mem.count.item() == 5
        retrieved = mem.read(patterns[0:1])
        assert retrieved.shape == (1, 64)

    def test_novelty_computation(self):
        mem = EpisodicMemory(capacity=100, d_memory=32)
        patterns = torch.randn(5, 32)
        mem.write(patterns)
        novelty = mem.compute_novelty(patterns[0:1])
        assert novelty.shape == (1,)
        assert 0 <= novelty.item() <= 1

    def test_novelty_weighted_write(self):
        mem = EpisodicMemory(capacity=5, d_memory=32)
        for _ in range(5):
            mem.write(torch.randn(1, 32))
        assert mem.count.item() == 5
        mem.write(torch.randn(1, 32), novelty=torch.tensor([0.99]))
        assert mem.count.item() == 5

    def test_top_k_read(self):
        mem = EpisodicMemory(capacity=100, d_memory=32)
        mem.write(torch.randn(20, 32))
        retrieved = mem.read(torch.randn(2, 32), top_k=5)
        assert retrieved.shape == (2, 32)

    def test_empty_read(self):
        mem = EpisodicMemory(capacity=10, d_memory=32)
        retrieved = mem.read(torch.randn(2, 32))
        assert (retrieved == 0).all()

    def test_reset(self):
        mem = EpisodicMemory(capacity=10, d_memory=32)
        mem.write(torch.randn(5, 32))
        mem.reset()
        assert mem.count.item() == 0

    def test_statistics(self):
        mem = EpisodicMemory(capacity=100, d_memory=32)
        mem.write(torch.randn(10, 32))
        stats = mem.get_statistics()
        assert stats["n_stored"] == 10
        assert stats["utilization"] == 0.1


class TestEWCRegularizer:
    def test_penalty_zero_initially(self):
        model = nn.Linear(32, 32)
        ewc = EWCRegularizer(model, lambda_ewc=100.0)
        assert ewc.penalty().item() == 0.0

    def test_penalty_after_change(self):
        model = nn.Linear(32, 32)
        ewc = EWCRegularizer(model, lambda_ewc=100.0)
        for name, param in model.named_parameters():
            if param.requires_grad:
                ewc.fisher[name] = torch.ones_like(param.data)
                ewc.reference_params[name] = param.data.clone()
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.1)
        assert ewc.penalty().item() > 0


class TestSemanticConsolidation:
    def test_empty_buffer(self):
        sc = SemanticConsolidation(d_memory=32)
        buf = ReplayBuffer(capacity=100)
        metrics = sc.consolidate(buf, nn.Linear(32, 32))
        assert metrics["consolidation_loss"] == 0.0

    def test_with_experiences(self):
        sc = SemanticConsolidation(d_memory=32)
        buf = ReplayBuffer(capacity=100)
        for _ in range(10):
            buf.add(Experience(states=torch.randn(8, 4, 32), alive=torch.ones(8, 4)))
        metrics = sc.consolidate(buf, nn.Linear(32, 32), batch_size=4)
        assert "consolidation_loss" in metrics


class TestConsolidationLoop:
    def test_store_experience(self):
        model = nn.Linear(32, 32)
        buf = ReplayBuffer(capacity=100)
        loop = ConsolidationLoop(model=model, replay_buffer=buf, consolidation_every=10, d_memory=32)
        loop.store_experience(states=torch.randn(8, 4, 32), alive=torch.ones(8, 4), prediction_error=0.5)
        assert len(buf) == 1

    def test_not_yet_triggered(self):
        model = nn.Linear(32, 32)
        buf = ReplayBuffer(capacity=100)
        loop = ConsolidationLoop(model=model, replay_buffer=buf, consolidation_every=100, d_memory=32)
        assert loop.maybe_consolidate() is None

    def test_triggers_at_interval(self):
        model = nn.Linear(32, 32)
        buf = ReplayBuffer(capacity=100)
        loop = ConsolidationLoop(model=model, replay_buffer=buf, consolidation_every=5, d_memory=32, batch_size=2)
        for _ in range(10):
            loop.store_experience(states=torch.randn(8, 4, 32), alive=torch.ones(8, 4))
        for _ in range(4):
            loop.maybe_consolidate()
        result = loop.maybe_consolidate()
        assert result is not None
        assert "consolidation_loss" in result


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.causal_graph = type("CG", (), {"adjacency": torch.rand(16, 16)})()

    def forward(self, x, **kwargs):
        B, T = x.shape[:2]
        class _Out:
            predicted_states = torch.randn(B, T, 16, 128)
            alive_mask = torch.ones(B, T, 16)
        return _Out()


class TestBenchmarks:
    @pytest.fixture
    def model(self):
        return _DummyModel()

    def test_physics_benchmark(self, model):
        result = PhysicsPredictionBenchmark(n_samples=3, resolution=64).evaluate(model)
        assert result.name == "physics_prediction"

    def test_causal_benchmark(self, model):
        result = CausalDiscoveryBenchmark(n_samples=3, n_objects=5).evaluate(model)
        assert result.name == "causal_discovery"

    def test_planning_benchmark(self, model):
        result = PlanningBenchmark(n_episodes=3).evaluate(model)
        assert "mean_goal_distance" in result.metrics

    def test_continual_benchmark(self, model):
        result = ContinualLearningBenchmark().evaluate(model)
        assert result.name == "continual_learning"

    def test_runner(self, model):
        runner = BenchmarkRunner(benchmarks=[PlanningBenchmark(n_episodes=2), ContinualLearningBenchmark()])
        results = runner.run_all(model)
        assert len(results) == 2
        assert "planning" in runner.results_table(results)
