"""Tests for gradient checkpointing (Sprint 2 Phase 2).

Verifies that gradient checkpointing flags are correctly set and that
forward/backward passes still work with checkpointing enabled.
"""

from __future__ import annotations

import torch
import pytest

from hclsm.config import HCLSMConfig


class TestGradientCheckpointingFlags:
    """Test that gradient checkpointing flags wire through model correctly."""

    @pytest.fixture
    def model(self):
        try:
            from hclsm.model import HCLSMWorldModel
        except ImportError:
            pytest.skip("einops not installed, skipping model tests")
        config = HCLSMConfig.tiny()
        return HCLSMWorldModel(config)

    def test_default_disabled(self, model):
        assert model.perception.use_gradient_checkpointing is False
        assert model.slot_attention.use_gradient_checkpointing is False
        assert model.level0_ssm.use_gradient_checkpointing is False
        assert model.level1_event.use_gradient_checkpointing is False

    def test_enable_sets_flags(self, model):
        model.enable_gradient_checkpointing()
        assert model.perception.use_gradient_checkpointing is True
        assert model.slot_attention.use_gradient_checkpointing is True
        assert model.level0_ssm.use_gradient_checkpointing is True
        assert model.level1_event.use_gradient_checkpointing is True

    def test_forward_backward_with_checkpointing(self, model):
        model.enable_gradient_checkpointing()
        video = torch.randn(1, 4, 3, 224, 224)
        output = model(video_frames=video, targets=video)
        loss = output.losses["total"]
        loss.backward()

        # Verify gradients exist
        has_grad = False
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                has_grad = True
                break
        assert has_grad, "No gradients computed with checkpointing enabled"

    def test_forward_backward_without_checkpointing(self, model):
        video = torch.randn(1, 4, 3, 224, 224)
        output = model(video_frames=video, targets=video)
        loss = output.losses["total"]
        loss.backward()

        has_grad = False
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                has_grad = True
                break
        assert has_grad, "No gradients computed without checkpointing"


class TestGradientCheckpointingConfig:
    """Test config integration for gradient checkpointing."""

    def test_config_flag_exists(self):
        config = HCLSMConfig.tiny()
        assert hasattr(config.training, "use_gradient_checkpointing")
        assert config.training.use_gradient_checkpointing is False

    def test_config_flag_set(self):
        config = HCLSMConfig.tiny()
        config.training.use_gradient_checkpointing = True
        assert config.training.use_gradient_checkpointing is True

    def test_config_roundtrip_yaml(self, tmp_path):
        config = HCLSMConfig.tiny()
        config.training.use_gradient_checkpointing = True
        path = tmp_path / "test_config.yaml"
        config.to_yaml(path)
        loaded = HCLSMConfig.from_yaml(path)
        assert loaded.training.use_gradient_checkpointing is True
