"""Tests for the hierarchical dynamics engine."""

from __future__ import annotations

import torch
import pytest

from hclsm.config import HCLSMConfig
from hclsm.dynamics.level0_ssm import Level0SSM
from hclsm.dynamics.event_detector import EventDetector
from hclsm.dynamics.level1_event import Level1EventDynamics
from hclsm.dynamics.level2_goal import Level2GoalDynamics
from hclsm.dynamics.hierarchy_manager import HierarchyManager


@pytest.fixture
def config():
    return HCLSMConfig.tiny()


class TestLevel0SSM:
    def test_output_shape(self, config):
        ssm = Level0SSM(config.dynamics.level0, config.objects.d_slot)
        B, N, T, D = 2, 16, 8, config.objects.d_slot
        x = torch.randn(B, N, T, D)
        mask = torch.ones(B, N)
        out = ssm(x, mask)
        assert out.shape == (B, N, T, D)

    def test_dead_objects_masked(self, config):
        ssm = Level0SSM(config.dynamics.level0, config.objects.d_slot)
        B, N, T, D = 2, 16, 4, config.objects.d_slot
        x = torch.randn(B, N, T, D)
        mask = torch.zeros(B, N)
        mask[:, :3] = 1.0
        out = ssm(x, mask)
        assert out[:, 3:].abs().sum() < 1e-5

    def test_gradient_flow(self, config):
        ssm = Level0SSM(config.dynamics.level0, config.objects.d_slot)
        x = torch.randn(1, 4, 4, config.objects.d_slot, requires_grad=True)
        mask = torch.ones(1, 4)
        out = ssm(x, mask)
        out.sum().backward()
        assert x.grad is not None


class TestEventDetector:
    def test_output_range(self, config):
        det = EventDetector(config.objects.d_slot)
        x = torch.randn(2, 8, config.objects.d_slot)
        probs, mask = det(x)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0
        assert probs.shape == (2, 8)

    def test_at_least_one_event(self, config):
        det = EventDetector(config.objects.d_slot, threshold=0.99)
        x = torch.zeros(2, 8, config.objects.d_slot)
        _, mask = det(x)
        assert mask.sum(dim=1).min() >= 1


class TestLevel1EventDynamics:
    def test_output_shape(self, config):
        l1 = Level1EventDynamics(config.dynamics.level1, config.objects.d_slot)
        B, K, N, D = 2, 3, 16, config.objects.d_slot
        x = torch.randn(B, K, N, D)
        mask = torch.ones(B, K)
        obj_mask = torch.ones(B, N)
        out = l1(x, mask, obj_mask)
        assert out.shape == (B, K, N, D)


class TestLevel2GoalDynamics:
    def test_returns_none_few_events(self, config):
        l2 = Level2GoalDynamics(config.dynamics.level2, config.objects.d_slot)
        x = torch.randn(2, 2, 16, config.objects.d_slot)  # K=2 < 4
        out = l2(x)
        assert out is None

    def test_output_shape_enough_events(self, config):
        l2 = Level2GoalDynamics(config.dynamics.level2, config.objects.d_slot)
        x = torch.randn(2, 6, 16, config.objects.d_slot)  # K=6 >= 4
        mask = torch.ones(2, 6)
        out = l2(x, mask)
        assert out is not None
        assert out.shape == (2, config.dynamics.level2.n_summary_tokens, config.dynamics.level2.d_model)


class TestHierarchyManager:
    def test_gather_scatter_roundtrip(self, config):
        hm = HierarchyManager(config.objects.d_slot, config.dynamics.level2.d_model)
        B, T, N, D = 2, 8, 16, config.objects.d_slot
        states = torch.randn(B, T, N, D)
        event_mask = torch.zeros(B, T)
        event_mask[:, 2] = 1.0
        event_mask[:, 5] = 1.0

        gathered, pad_mask, K = hm.gather_events(states, event_mask)
        assert gathered.shape[1] == K
        assert K == 2

    def test_gating_range(self, config):
        hm = HierarchyManager(config.objects.d_slot, config.dynamics.level2.d_model)
        B, T, N, D = 2, 8, 16, config.objects.d_slot
        l0 = torch.randn(B, T, N, D)
        l1 = torch.randn(B, 2, N, D)
        l2 = torch.randn(B, 8, config.dynamics.level2.d_model)
        event_mask = torch.zeros(B, T)
        event_mask[:, 0] = 1.0
        event_mask[:, 4] = 1.0

        out = hm(l0, l1, l2, event_mask)
        assert out.shape == (B, T, N, D)
        assert torch.isfinite(out).all()
