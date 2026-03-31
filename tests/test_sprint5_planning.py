"""Tests for Sprint 5: Level 2 + Planning."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from hclsm.config import Level2Config
from hclsm.dynamics.level2_goal import Level2GoalDynamics
from hclsm.causality.action_conditioner import ActionConditioner
from hclsm.causality.value_function import ValueFunction
from hclsm.inference.world_simulator import WorldSimulator, RolloutOutput
from hclsm.inference.planner import CEMPlanner, MPPIPlanner, PlanResult


# ═══════════════════════════════════════════════════════════════════════════
# Level2GoalDynamics
# ═══════════════════════════════════════════════════════════════════════════

class TestLevel2GoalDynamics:
    @pytest.fixture
    def l2_no_goal(self):
        config = Level2Config(d_model=64, n_layers=2, n_heads=4, n_summary_tokens=4)
        return Level2GoalDynamics(config, d_slot=32)

    @pytest.fixture
    def l2_with_goal(self):
        config = Level2Config(d_model=64, n_layers=2, n_heads=4, n_summary_tokens=4, d_goal=16)
        return Level2GoalDynamics(config, d_slot=32)

    def test_forward_no_goal(self, l2_no_goal):
        events = torch.randn(2, 8, 16, 32)
        pad_mask = torch.ones(2, 8)
        out = l2_no_goal(events, pad_mask)
        assert out is not None
        assert out.shape == (2, 4, 64)  # (B, n_summary, d_model)

    def test_too_few_events_returns_none(self, l2_no_goal):
        events = torch.randn(2, 2, 16, 32)  # Only 2 events < min_events=4
        out = l2_no_goal(events)
        assert out is None

    def test_forward_with_goal(self, l2_with_goal):
        events = torch.randn(2, 8, 16, 32)
        pad_mask = torch.ones(2, 8)
        goal = torch.randn(2, 16)
        out = l2_with_goal(events, pad_mask, goal=goal)
        assert out is not None
        assert out.shape == (2, 4, 64)

    def test_backward_compatible_no_goal(self, l2_with_goal):
        """Goal-conditioned model works without goal (backward compatible)."""
        events = torch.randn(2, 8, 16, 32)
        pad_mask = torch.ones(2, 8)
        out = l2_with_goal(events, pad_mask)  # No goal
        assert out is not None

    def test_decompress(self, l2_no_goal):
        events = torch.randn(2, 8, 16, 32)
        abstract = l2_no_goal(events)
        assert abstract is not None
        decompressed = l2_no_goal.decompress_to_slot_dim(abstract)
        assert decompressed.shape == (2, 32)  # (B, d_slot)

    def test_gradient_flow(self, l2_with_goal):
        events = torch.randn(2, 8, 16, 32)
        goal = torch.randn(2, 16, requires_grad=True)
        out = l2_with_goal(events, goal=goal)
        out.sum().backward()
        assert goal.grad is not None


# ═══════════════════════════════════════════════════════════════════════════
# WorldSimulator
# ═══════════════════════════════════════════════════════════════════════════

class _DummyModel(nn.Module):
    """Minimal model mock for testing WorldSimulator."""

    def __init__(self, n_slots=8, d_slot=32):
        super().__init__()
        self.n_slots = n_slots
        self.d_slot = d_slot
        self.linear = nn.Linear(d_slot, d_slot)

    def forward(self, video_frames, **kwargs):
        B, T = video_frames.shape[:2]
        states = torch.randn(B, T, self.n_slots, self.d_slot)
        alive = torch.ones(B, T, self.n_slots)

        class _Out:
            predicted_states = states
            alive_mask = alive
            event_mask = None

        return _Out()


class TestWorldSimulator:
    @pytest.fixture
    def simulator(self):
        model = _DummyModel(n_slots=8, d_slot=32)
        ac = ActionConditioner(d_action=4, d_slot=32, n_max_slots=8)
        return WorldSimulator(model, action_conditioner=ac)

    def test_rollout_no_actions(self, simulator):
        frames = torch.randn(2, 4, 3, 64, 64)
        result = simulator.rollout(frames, n_steps=5)
        assert isinstance(result, RolloutOutput)
        assert result.predicted_states.shape == (2, 5, 8, 32)
        assert result.alive_masks.shape == (2, 5, 8)

    def test_rollout_with_actions(self, simulator):
        frames = torch.randn(2, 4, 3, 64, 64)
        actions = torch.randn(2, 5, 4)
        result = simulator.rollout(frames, n_steps=5, actions=actions)
        assert result.predicted_states.shape == (2, 5, 8, 32)

    def test_batched_rollout(self, simulator):
        states = torch.randn(2, 8, 32)
        alive = torch.ones(2, 8)
        actions = torch.randn(2, 16, 5, 4)  # B=2, K=16, T=5, d_action=4
        result = simulator.batched_rollout(states, alive, actions)
        assert result.predicted_states.shape == (32, 5, 8, 32)  # B*K=32

    def test_evaluate_trajectory_goal(self, simulator):
        traj = torch.randn(2, 5, 8, 32)
        alive = torch.ones(2, 5, 8)
        goal = torch.randn(2, 8, 32)
        costs = simulator.evaluate_trajectory(traj, alive, goal_states=goal)
        assert costs.shape == (2,)
        assert (costs >= 0).all()

    def test_evaluate_trajectory_value_fn(self, simulator):
        traj = torch.randn(2, 5, 8, 32)
        alive = torch.ones(2, 5, 8)
        vf = ValueFunction(d_slot=32)
        costs = simulator.evaluate_trajectory(traj, alive, value_fn=vf)
        assert costs.shape == (2,)

    def test_evaluate_trajectory_default(self, simulator):
        traj = torch.randn(2, 5, 8, 32)
        alive = torch.ones(2, 5, 8)
        costs = simulator.evaluate_trajectory(traj, alive)
        assert costs.shape == (2,)


# ═══════════════════════════════════════════════════════════════════════════
# CEM Planner
# ═══════════════════════════════════════════════════════════════════════════

class TestCEMPlanner:
    @pytest.fixture
    def planner(self):
        model = _DummyModel(n_slots=8, d_slot=32)
        ac = ActionConditioner(d_action=4, d_slot=32, n_max_slots=8)
        sim = WorldSimulator(model, action_conditioner=ac)
        return CEMPlanner(
            simulator=sim, horizon=5, n_samples=32,
            n_elites=8, n_iterations=3, d_action=4,
        )

    def test_plan_with_goal(self, planner):
        states = torch.randn(2, 8, 32)
        alive = torch.ones(2, 8)
        goal = torch.randn(2, 8, 32)
        result = planner.plan(states, alive, goal_states=goal)
        assert isinstance(result, PlanResult)
        assert result.action.shape == (2, 4)
        assert result.action_sequence.shape == (2, 5, 4)
        assert result.best_cost.shape == (2,)

    def test_plan_action_bounds(self, planner):
        states = torch.randn(2, 8, 32)
        alive = torch.ones(2, 8)
        goal = torch.randn(2, 8, 32)
        result = planner.plan(states, alive, goal_states=goal)
        assert result.action_sequence.min() >= -1.0
        assert result.action_sequence.max() <= 1.0

    def test_plan_with_value_fn(self, planner):
        states = torch.randn(2, 8, 32)
        alive = torch.ones(2, 8)
        vf = ValueFunction(d_slot=32)
        result = planner.plan(states, alive, value_fn=vf)
        assert result.action.shape == (2, 4)


# ═══════════════════════════════════════════════════════════════════════════
# MPPI Planner
# ═══════════════════════════════════════════════════════════════════════════

class TestMPPIPlanner:
    @pytest.fixture
    def planner(self):
        model = _DummyModel(n_slots=8, d_slot=32)
        ac = ActionConditioner(d_action=4, d_slot=32, n_max_slots=8)
        sim = WorldSimulator(model, action_conditioner=ac)
        return MPPIPlanner(
            simulator=sim, horizon=5, n_samples=32,
            temperature=1.0, d_action=4,
        )

    def test_plan_with_goal(self, planner):
        states = torch.randn(2, 8, 32)
        alive = torch.ones(2, 8)
        goal = torch.randn(2, 8, 32)
        result = planner.plan(states, alive, goal_states=goal)
        assert isinstance(result, PlanResult)
        assert result.action.shape == (2, 4)
        assert result.action_sequence.shape == (2, 5, 4)

    def test_warm_start(self, planner):
        states = torch.randn(2, 8, 32)
        alive = torch.ones(2, 8)
        goal = torch.randn(2, 8, 32)

        # First plan
        result1 = planner.plan(states, alive, goal_states=goal)
        assert planner._prev_plan is not None

        # Second plan should warm-start from previous
        result2 = planner.plan(states, alive, goal_states=goal)
        assert result2.action.shape == (2, 4)

    def test_reset(self, planner):
        states = torch.randn(2, 8, 32)
        alive = torch.ones(2, 8)
        planner.plan(states, alive)
        assert planner._prev_plan is not None
        planner.reset()
        assert planner._prev_plan is None

    def test_action_bounds(self, planner):
        states = torch.randn(2, 8, 32)
        alive = torch.ones(2, 8)
        result = planner.plan(states, alive)
        assert result.action_sequence.min() >= -1.0
        assert result.action_sequence.max() <= 1.0
