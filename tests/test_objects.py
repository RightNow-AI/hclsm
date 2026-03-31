"""Tests for the object decomposition layer."""

from __future__ import annotations

import torch
import pytest

from hclsm.config import HCLSMConfig
from hclsm.objects.dynamic_slots import DynamicSlotAttention
from hclsm.objects.slot_tracker import SlotTracker
from hclsm.objects.relation_graph import RelationGraph
from hclsm.objects.losses import slot_diversity_loss, slot_tracking_loss


@pytest.fixture
def config():
    return HCLSMConfig.tiny()


class TestDynamicSlotAttention:
    def test_output_shapes(self, config):
        sa = DynamicSlotAttention(config.objects, config.d_world)
        inputs = torch.randn(4, 196, config.d_world)
        slots, alive = sa(inputs)
        assert slots.shape == (4, config.objects.n_max_slots, config.objects.d_slot)
        assert alive.shape == (4, config.objects.n_max_slots)

    def test_alive_in_range(self, config):
        sa = DynamicSlotAttention(config.objects, config.d_world)
        inputs = torch.randn(2, 196, config.d_world)
        _, alive = sa(inputs)
        assert alive.min() >= 0.0
        assert alive.max() <= 1.0

    def test_gradient_flow(self, config):
        sa = DynamicSlotAttention(config.objects, config.d_world)
        inputs = torch.randn(2, 50, config.d_world)
        slots, alive = sa(inputs)
        loss = slots.sum() + alive.sum()
        loss.backward()
        assert sa.slot_mu.grad is not None


class TestSlotTracker:
    def test_permutation_valid(self, config):
        tracker = SlotTracker(config.objects)
        N = config.objects.n_max_slots
        D = config.objects.d_slot
        s1 = torch.randn(2, N, D)
        s2 = torch.randn(2, N, D)
        alive = torch.ones(2, N)
        perm, matched = tracker(s1, s2, alive, alive)
        assert perm.shape == (2, N)
        # Permutation should contain valid indices
        assert perm.min() >= 0
        assert perm.max() < N


class TestRelationGraph:
    def test_output_shapes(self, config):
        gnn = RelationGraph(config.objects)
        N = config.objects.n_max_slots
        D = config.objects.d_slot
        nodes = torch.randn(2, N, D)
        alive = torch.ones(2, N)
        updated, causal = gnn(nodes, alive)
        assert updated.shape == (2, N, D)
        assert causal.shape == (2, N, N)

    def test_dead_slots_masked(self, config):
        gnn = RelationGraph(config.objects)
        N = config.objects.n_max_slots
        D = config.objects.d_slot
        nodes = torch.randn(2, N, D)
        alive = torch.zeros(2, N)
        alive[:, :3] = 1.0  # Only 3 alive
        updated, _ = gnn(nodes, alive)
        # Dead slots should be zeroed
        assert (updated[:, 3:].abs().sum() < 1e-5)


class TestObjectLosses:
    def test_diversity_loss(self):
        slots = torch.randn(4, 8, 64)
        alive = torch.ones(4, 8)
        loss = slot_diversity_loss(slots, alive)
        assert torch.isfinite(loss)
        assert loss >= 0

    def test_tracking_loss(self):
        slots = torch.randn(2, 10, 8, 64)
        alive = torch.ones(2, 10, 8)
        loss = slot_tracking_loss(slots, alive)
        assert torch.isfinite(loss)
        assert loss >= 0
