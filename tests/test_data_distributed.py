"""Tests for Sprint 2 data pipeline and distributed utilities."""

from __future__ import annotations

import torch
import pytest

from hclsm.config import HCLSMConfig
from hclsm.training.data import (
    SyntheticVideoDataset,
    build_dataloader,
    build_train_transforms,
    build_eval_transforms,
    TemporalSubsample,
    RandomTemporalCrop,
    SpatialResize,
    RandomSpatialCrop,
    RandomHorizontalFlipVideo,
    NormalizeVideo,
    Compose,
)
from hclsm.training.distributed import (
    is_main_process,
    get_rank,
    get_world_size,
    wrap_model_distributed,
)


# ═══════════════════════════════════════════════════════════════════════════
# Augmentations
# ═══════════════════════════════════════════════════════════════════════════

class TestTemporalSubsample:
    def test_subsample(self):
        video = torch.rand(32, 3, 64, 64)
        t = TemporalSubsample(16)
        out = t(video)
        assert out.shape == (16, 3, 64, 64)

    def test_pad_short_video(self):
        video = torch.rand(4, 3, 64, 64)
        t = TemporalSubsample(16)
        out = t(video)
        assert out.shape == (16, 3, 64, 64)

    def test_exact_length(self):
        video = torch.rand(16, 3, 64, 64)
        t = TemporalSubsample(16)
        out = t(video)
        assert out.shape == (16, 3, 64, 64)


class TestRandomTemporalCrop:
    def test_crop(self):
        video = torch.rand(32, 3, 64, 64)
        t = RandomTemporalCrop(16)
        out = t(video)
        assert out.shape == (16, 3, 64, 64)

    def test_pad_short(self):
        video = torch.rand(4, 3, 64, 64)
        t = RandomTemporalCrop(16)
        out = t(video)
        assert out.shape == (16, 3, 64, 64)


class TestSpatialResize:
    def test_resize(self):
        video = torch.rand(8, 3, 320, 240)
        t = SpatialResize(224)
        out = t(video)
        assert out.shape == (8, 3, 224, 224)

    def test_noop_same_size(self):
        video = torch.rand(8, 3, 224, 224)
        t = SpatialResize(224)
        out = t(video)
        assert out.shape == (8, 3, 224, 224)
        assert torch.equal(out, video)


class TestRandomSpatialCrop:
    def test_crop(self):
        video = torch.rand(8, 3, 320, 320)
        t = RandomSpatialCrop(224)
        out = t(video)
        assert out.shape == (8, 3, 224, 224)

    def test_upscale_then_crop(self):
        video = torch.rand(8, 3, 100, 100)
        t = RandomSpatialCrop(224)
        out = t(video)
        assert out.shape == (8, 3, 224, 224)


class TestRandomHorizontalFlip:
    def test_flip_preserves_shape(self):
        video = torch.rand(8, 3, 64, 64)
        t = RandomHorizontalFlipVideo(p=1.0)
        out = t(video)
        assert out.shape == video.shape

    def test_no_flip(self):
        video = torch.rand(8, 3, 64, 64)
        t = RandomHorizontalFlipVideo(p=0.0)
        out = t(video)
        assert torch.equal(out, video)


class TestNormalize:
    def test_normalize(self):
        video = torch.rand(8, 3, 64, 64)
        t = NormalizeVideo()
        out = t(video)
        assert out.shape == video.shape
        # After normalization, values shift from [0,1] range
        assert out.mean().abs() < 2.0  # reasonable sanity check


class TestCompose:
    def test_chain(self):
        video = torch.rand(32, 3, 320, 240)
        t = Compose([TemporalSubsample(16), SpatialResize(224)])
        out = t(video)
        assert out.shape == (16, 3, 224, 224)


class TestBuildTransforms:
    def test_train_transforms(self):
        t = build_train_transforms(n_frames=8, resolution=128)
        video = torch.rand(32, 3, 256, 256)
        out = t(video)
        assert out.shape == (8, 3, 128, 128)

    def test_eval_transforms(self):
        t = build_eval_transforms(n_frames=8, resolution=128)
        video = torch.rand(32, 3, 256, 256)
        out = t(video)
        assert out.shape == (8, 3, 128, 128)


# ═══════════════════════════════════════════════════════════════════════════
# build_dataloader
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildDataloader:
    def test_synthetic_default(self):
        config = HCLSMConfig.tiny()
        loader = build_dataloader(config, split="train")
        batch = next(iter(loader))
        assert "video" in batch
        T = config.perception.temporal_resolution
        H = config.perception.input_resolution
        assert batch["video"].shape[1] == T
        assert batch["video"].shape[2] == 3
        assert batch["video"].shape[3] == H
        assert batch["video"].shape[4] == H

    def test_synthetic_val(self):
        config = HCLSMConfig.tiny()
        loader = build_dataloader(config, split="val")
        batch = next(iter(loader))
        assert "video" in batch

    def test_fallback_for_missing_data_dir(self):
        config = HCLSMConfig.tiny()
        config.training.dataset_name = "video"
        config.training.data_dir = "/nonexistent/path"
        loader = build_dataloader(config, split="train")
        batch = next(iter(loader))
        assert "video" in batch  # should fallback to synthetic


# ═══════════════════════════════════════════════════════════════════════════
# Distributed utilities (non-distributed tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestDistributedUtils:
    def test_is_main_process(self):
        assert is_main_process() is True

    def test_get_rank(self):
        assert get_rank() == 0

    def test_get_world_size(self):
        assert get_world_size() == 1

    def test_wrap_model_noop_no_config(self):
        model = torch.nn.Linear(10, 10)
        wrapped = wrap_model_distributed(model)
        assert wrapped is model

    def test_wrap_model_noop_fsdp_disabled(self):
        config = HCLSMConfig.tiny()
        config.training.fsdp_enabled = False
        model = torch.nn.Linear(10, 10)
        wrapped = wrap_model_distributed(model, config)
        assert wrapped is model

    def test_wrap_model_noop_fsdp_no_dist(self):
        config = HCLSMConfig.tiny()
        config.training.fsdp_enabled = True
        model = torch.nn.Linear(10, 10)
        wrapped = wrap_model_distributed(model, config)
        assert wrapped is model  # no process group, so no-op
