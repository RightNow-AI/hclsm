"""Tests for the perception encoder."""

from __future__ import annotations

import torch
import pytest

from hclsm.config import HCLSMConfig, PerceptionConfig
from hclsm.perception.tokenizer import PatchEmbedding
from hclsm.perception.vision_encoder import VisionEncoder
from hclsm.perception.multimodal_fuser import MultimodalFuser


@pytest.fixture
def config():
    return HCLSMConfig.tiny()


class TestPatchEmbedding:
    def test_output_shape(self, config):
        embed = PatchEmbedding(config.perception)
        x = torch.randn(2, 4, 3, 224, 224)
        out = embed(x)
        assert out.shape == (2, 4, 196, config.perception.d_model)

    def test_n_patches(self, config):
        embed = PatchEmbedding(config.perception)
        assert embed.n_patches == 196


class TestVisionEncoder:
    def test_output_shape(self, config):
        encoder = VisionEncoder(config.perception)
        x = torch.randn(2, 4, 3, 224, 224)
        out = encoder(x)
        assert out.shape == (2, 4, 196, config.perception.d_model)

    def test_gradient_flow(self, config):
        encoder = VisionEncoder(config.perception)
        x = torch.randn(1, 2, 3, 224, 224, requires_grad=True)
        out = encoder(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestMultimodalFuser:
    def test_vision_only(self, config):
        fuser = MultimodalFuser(config)
        patch_embed = torch.randn(2, 4, 196, config.perception.d_model)
        out = fuser({"vision": patch_embed})
        assert out.shape == (2, 4, 196, config.d_world)
