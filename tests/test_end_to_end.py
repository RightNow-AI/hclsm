"""End-to-end tests for HCLSMWorldModel.

CRITICAL: These tests verify the complete forward pass, gradient flow,
loss computation, and EMA target encoder.
"""

from __future__ import annotations

import torch
import pytest

from hclsm.config import HCLSMConfig
from hclsm.model import HCLSMWorldModel


@pytest.fixture
def tiny_config():
    return HCLSMConfig.tiny()


@pytest.fixture
def model(tiny_config):
    return HCLSMWorldModel(tiny_config)


def _make_video(B: int, T: int, config: HCLSMConfig) -> torch.Tensor:
    H = config.perception.input_resolution
    return torch.randn(B, T, 3, H, H)


class TestForwardPass:
    def test_forward_shapes(self, model, tiny_config):
        """Forward pass produces correct output shapes."""
        B, T = 2, 8
        video = _make_video(B, T, tiny_config)
        output = model(video)

        N = tiny_config.objects.n_max_slots
        d = tiny_config.objects.d_slot

        assert output.predicted_states.shape == (B, T, N, d)
        assert output.object_slots.shape == (B, T, N, d)
        assert output.alive_mask.shape == (B, T, N)
        assert output.event_mask.shape == (B, T)
        assert output.event_scores.shape == (B, T)
        assert output.edge_weights.shape == (B, T, N, N)

    def test_forward_no_targets(self, model, tiny_config):
        """Forward pass works without targets (inference mode)."""
        video = _make_video(2, 4, tiny_config)
        output = model(video)
        assert len(output.losses) == 0
        assert output.predicted_states is not None


class TestLosses:
    def test_losses_finite(self, model, tiny_config):
        """All losses are finite."""
        video = _make_video(2, 8, tiny_config)
        output = model(video, targets=video)

        for name, val in output.losses.items():
            assert torch.isfinite(val), f"Loss {name} is not finite: {val}"

    def test_losses_computed(self, model, tiny_config):
        """Key losses are present and non-negative."""
        video = _make_video(2, 8, tiny_config)
        output = model(video, targets=video)

        assert "total" in output.losses
        assert "prediction" in output.losses
        assert "diversity" in output.losses
        assert "tracking" in output.losses
        assert output.losses["total"] >= 0


class TestGradients:
    def test_gradient_flow(self, model, tiny_config):
        """Gradients flow to core trainable parameters.

        Some params are expected to have no gradient in Sprint 1:
        - birth_proj: used inside torch.no_grad() block
        - slot_tracker: uses scipy (non-differentiable)
        - causal_mlp: causal graph disabled in tiny config
        - position/velocity_head: not used in loss yet
        - event_detector: straight-through, may not flow grad
        - level2_goal: skipped when K < min_events
        - hierarchy_manager L2 gates: skipped when L2 is None
        - loss_fn compression MLPs: depend on L1/L2 being active
        """
        video = _make_video(2, 8, tiny_config)
        output = model(video, targets=video)
        output.losses["total"].backward()

        # Known-dead prefixes in Sprint 1 tiny config
        known_dead_prefixes = (
            "slot_attention.birth_proj",
            "slot_tracker.",
            "relation_graph.causal_mlp",
            "position_head",
            "velocity_head",
            "event_detector",
            "level2_goal",
            "hierarchy_manager.l2_to_l1",
            "hierarchy_manager.gate_l2_l0",
            "hierarchy_manager.gate_l0",
            "loss_fn.compress_l1_to_l2",
        )

        unexpected_dead = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
                if not name.startswith(known_dead_prefixes):
                    unexpected_dead.append(name)

        assert len(unexpected_dead) == 0, (
            f"Unexpected dead parameters: {unexpected_dead}"
        )

    def test_no_nan_gradients(self, model, tiny_config):
        """Core module gradients should be finite."""
        torch.manual_seed(42)
        video = _make_video(2, 8, tiny_config)
        output = model(video, targets=video)
        output.losses["total"].backward()

        # Check core modules have finite gradients
        core_prefixes = ("perception.", "fuser.", "slot_attention.", "level0_ssm.",
                         "level1_event.", "relation_graph.edge_mlp",
                         "relation_graph.msg_mlp", "relation_graph.update_mlp")
        nan_params = []
        for name, param in model.named_parameters():
            if param.grad is not None and name.startswith(core_prefixes):
                if not torch.isfinite(param.grad).all():
                    nan_params.append(name)

        assert len(nan_params) == 0, f"NaN gradients in core params: {nan_params}"


class TestEMA:
    def test_ema_update(self, model, tiny_config):
        """EMA target encoder updates correctly."""
        # Save initial target params
        target_before = {
            n: p.clone()
            for n, p in model.target_perception.named_parameters()
        }

        # Manually perturb online encoder to simulate an optimizer step
        with torch.no_grad():
            for p in model.perception.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        model.update_target_encoder(decay=0.996)

        # Verify target params changed
        changed = False
        for name, param in model.target_perception.named_parameters():
            if not torch.equal(param, target_before[name]):
                changed = True
                break

        assert changed, "Target encoder params did not change after EMA update"


class TestVariableSlots:
    def test_alive_mask_varies(self, model, tiny_config):
        """Alive mask should not be uniform across all samples."""
        video = _make_video(4, 8, tiny_config)
        output = model(video)
        alive = output.alive_mask

        # Check that alive values are not all identical
        assert alive.std() > 1e-4, (
            "All alive values identical — existence head not working"
        )

    def test_alive_in_range(self, model, tiny_config):
        """Alive probabilities should be in [0, 1]."""
        video = _make_video(2, 4, tiny_config)
        output = model(video)
        assert output.alive_mask.min() >= 0.0
        assert output.alive_mask.max() <= 1.0
