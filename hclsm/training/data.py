"""Data loading for HCLSM training.

Sprint 2: WebDataset support, real video decoding (decord), temporal/spatial
augmentations, and a build_dataloader() factory driven by TrainingConfig.
"""

from __future__ import annotations

import glob
import logging
import os
import random
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports — graceful fallback when not installed
# ---------------------------------------------------------------------------
try:
    import decord
    decord.bridge.set_bridge("torch")
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

try:
    import webdataset as wds
    WEBDATASET_AVAILABLE = True
except ImportError:
    WEBDATASET_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# Video augmentations
# ═══════════════════════════════════════════════════════════════════════════

class TemporalSubsample:
    """Uniformly sample `n_frames` from the temporal axis."""

    def __init__(self, n_frames: int) -> None:
        self.n_frames = n_frames

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """Args: video (T, C, H, W). Returns: (n_frames, C, H, W)."""
        T = video.shape[0]
        if T <= self.n_frames:
            # Pad by repeating last frame
            pad = self.n_frames - T
            video = torch.cat([video, video[-1:].expand(pad, -1, -1, -1)], dim=0)
            return video
        indices = torch.linspace(0, T - 1, self.n_frames).long()
        return video[indices]


class RandomTemporalCrop:
    """Randomly crop a contiguous window of `n_frames` from the video."""

    def __init__(self, n_frames: int) -> None:
        self.n_frames = n_frames

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        T = video.shape[0]
        if T <= self.n_frames:
            pad = self.n_frames - T
            video = torch.cat([video, video[-1:].expand(pad, -1, -1, -1)], dim=0)
            return video
        start = random.randint(0, T - self.n_frames)
        return video[start : start + self.n_frames]


class SpatialResize:
    """Resize spatial dimensions to (resolution, resolution)."""

    def __init__(self, resolution: int) -> None:
        self.resolution = resolution

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """Args: video (T, C, H, W). Returns: (T, C, res, res)."""
        if video.shape[2] == self.resolution and video.shape[3] == self.resolution:
            return video
        return F.interpolate(
            video, size=(self.resolution, self.resolution),
            mode="bilinear", align_corners=False,
        )


class RandomSpatialCrop:
    """Random square crop from the spatial dimensions."""

    def __init__(self, resolution: int) -> None:
        self.resolution = resolution

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        _, _, H, W = video.shape
        if H == self.resolution and W == self.resolution:
            return video
        if H < self.resolution or W < self.resolution:
            video = F.interpolate(
                video, size=(max(H, self.resolution), max(W, self.resolution)),
                mode="bilinear", align_corners=False,
            )
            _, _, H, W = video.shape
        top = random.randint(0, H - self.resolution)
        left = random.randint(0, W - self.resolution)
        return video[:, :, top : top + self.resolution, left : left + self.resolution]


class RandomHorizontalFlipVideo:
    """Flip all frames horizontally with probability p."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return video.flip(-1)
        return video


class NormalizeVideo:
    """Per-channel normalization using ImageNet stats."""

    def __init__(
        self,
        mean: tuple[float, ...] = (0.485, 0.456, 0.406),
        std: tuple[float, ...] = (0.229, 0.224, 0.225),
    ) -> None:
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        return (video - self.mean.to(video.device)) / self.std.to(video.device)


class Compose:
    """Chain multiple video transforms."""

    def __init__(self, transforms: list[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            video = t(video)
        return video


def build_train_transforms(
    n_frames: int = 16, resolution: int = 224,
) -> Compose:
    """Default training augmentations."""
    return Compose([
        RandomTemporalCrop(n_frames),
        RandomSpatialCrop(resolution),
        RandomHorizontalFlipVideo(p=0.5),
        NormalizeVideo(),
    ])


def build_eval_transforms(
    n_frames: int = 16, resolution: int = 224,
) -> Compose:
    """Default eval augmentations (deterministic)."""
    return Compose([
        TemporalSubsample(n_frames),
        SpatialResize(resolution),
        NormalizeVideo(),
    ])


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic dataset (kept from Sprint 1 — used in tests & debugging)
# ═══════════════════════════════════════════════════════════════════════════

class SyntheticVideoDataset(Dataset):
    """Synthetic video dataset with moving colored rectangles.

    Generates simple scenes for testing slot attention decomposition.
    Each scene has 2-5 colored rectangles with linear motion.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        n_frames: int = 16,
        resolution: int = 224,
        n_objects_range: tuple[int, int] = (2, 5),
    ) -> None:
        self.n_samples = n_samples
        self.n_frames = n_frames
        self.resolution = resolution
        self.n_objects_range = n_objects_range

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Generate a synthetic video with moving objects.

        Returns:
            Dict with 'video': (T, C, H, W) float tensor in [0, 1].
        """
        T, H, W = self.n_frames, self.resolution, self.resolution
        video = torch.zeros(T, 3, H, W)

        n_obj = torch.randint(
            self.n_objects_range[0], self.n_objects_range[1] + 1, (1,)
        ).item()

        for _ in range(n_obj):
            color = torch.rand(3)
            obj_h = torch.randint(20, 60, (1,)).item()
            obj_w = torch.randint(20, 60, (1,)).item()
            y0 = torch.randint(0, max(1, H - obj_h), (1,)).item()
            x0 = torch.randint(0, max(1, W - obj_w), (1,)).item()
            vy = torch.randint(-3, 4, (1,)).item()
            vx = torch.randint(-3, 4, (1,)).item()

            for t in range(T):
                y = max(0, min(H - obj_h, y0 + vy * t))
                x = max(0, min(W - obj_w, x0 + vx * t))
                video[t, :, y:y + obj_h, x:x + obj_w] = color[:, None, None]

        return {"video": video}


# ═══════════════════════════════════════════════════════════════════════════
# Real video dataset (decord backend)
# ═══════════════════════════════════════════════════════════════════════════

class VideoDataset(Dataset):
    """Real video dataset backed by decord for fast decoding.

    Reads video files from disk, decodes with decord, applies augmentations.
    Falls back to synthetic data when decord is unavailable.
    """

    EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    def __init__(
        self,
        video_paths: list[str],
        n_frames: int = 16,
        resolution: int = 224,
        transform: Callable | None = None,
    ) -> None:
        self.video_paths = video_paths
        self.n_frames = n_frames
        self.resolution = resolution
        self.transform = transform or build_train_transforms(n_frames, resolution)

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Load and preprocess a video.

        Returns:
            Dict with 'video': (T, C, H, W) normalized tensor.
        """
        if not DECORD_AVAILABLE:
            logger.warning("decord not available, returning synthetic data")
            return SyntheticVideoDataset(
                n_samples=1, n_frames=self.n_frames, resolution=self.resolution,
            )[0]

        path = self.video_paths[idx]
        vr = decord.VideoReader(path, num_threads=1)
        total_frames = len(vr)

        # Sample frame indices
        if total_frames >= self.n_frames:
            indices = torch.linspace(0, total_frames - 1, self.n_frames).long().tolist()
        else:
            indices = list(range(total_frames))

        # Decode — decord with torch bridge returns (T, H, W, C) uint8
        frames = vr.get_batch(indices)  # (T, H, W, C) torch uint8
        video = frames.permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)

        video = self.transform(video)
        return {"video": video}

    @classmethod
    def from_directory(
        cls,
        data_dir: str,
        n_frames: int = 16,
        resolution: int = 224,
        transform: Callable | None = None,
    ) -> VideoDataset:
        """Scan a directory recursively for video files."""
        paths: list[str] = []
        for ext in cls.EXTENSIONS:
            paths.extend(glob.glob(os.path.join(data_dir, "**", f"*{ext}"), recursive=True))
        paths.sort()
        if not paths:
            raise FileNotFoundError(
                f"No video files ({', '.join(cls.EXTENSIONS)}) found in {data_dir}"
            )
        logger.info(f"Found {len(paths)} videos in {data_dir}")
        return cls(paths, n_frames=n_frames, resolution=resolution, transform=transform)


# ═══════════════════════════════════════════════════════════════════════════
# OpenX-Embodiment via LeRobot
# ═══════════════════════════════════════════════════════════════════════════

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset as _LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False


class OpenXDataset(Dataset):
    """Open X-Embodiment dataset via LeRobot format.

    Wraps LeRobot's HuggingFace-backed dataset for robot video + action pairs.
    Each sample yields consecutive frames from a single episode.

    Returns:
        Dict with 'video': (T, C, H, W) and 'action': (T, d_action).
    """

    def __init__(
        self,
        repo_id: str = "lerobot/pusht",
        n_frames: int = 16,
        resolution: int = 224,
        video_backend: str = "pyav",
    ) -> None:
        if not LEROBOT_AVAILABLE:
            raise ImportError("lerobot required: pip install lerobot")

        self.n_frames = n_frames
        self.resolution = resolution

        self._ds = _LeRobotDataset(repo_id, video_backend=video_backend)

        # Discover image and action keys from first sample
        sample0 = self._ds[0]
        self._image_key = None
        for k in sample0:
            if k.startswith("observation.image") or k.startswith("observation.images"):
                v = sample0[k]
                if hasattr(v, "shape") and len(v.shape) >= 2:
                    self._image_key = k
                    break
        if self._image_key is None:
            raise ValueError(f"No image key found in dataset. Keys: {list(sample0.keys())}")

        self._has_action = "action" in sample0
        if self._has_action:
            self._d_action = sample0["action"].shape[-1]
        else:
            self._d_action = 0

        # Build episode boundaries for temporal sampling
        self._episode_starts: list[int] = [0]
        self._episode_lengths: list[int] = []
        last_ep = -1
        for i in range(len(self._ds)):
            ep = self._ds[i]["episode_index"].item()
            if ep != last_ep and last_ep >= 0:
                self._episode_starts.append(i)
                self._episode_lengths.append(i - self._episode_starts[-2])
            last_ep = ep
        self._episode_lengths.append(len(self._ds) - self._episode_starts[-1])
        self._n_episodes = len(self._episode_starts)

        logger.info(
            f"OpenXDataset: {repo_id} | {len(self._ds)} frames, "
            f"{self._n_episodes} episodes | image={self._image_key} "
            f"d_action={self._d_action}"
        )

    @property
    def d_action(self) -> int:
        return self._d_action

    def __len__(self) -> int:
        # Multiply by 100 so the dataloader doesn't exhaust in one epoch
        # __getitem__ uses modulo so this cycles safely
        return self._n_episodes * 100

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_start = self._episode_starts[idx % self._n_episodes]
        ep_len = self._episode_lengths[idx % self._n_episodes]

        # Sample n_frames consecutive frames from this episode
        if ep_len <= self.n_frames:
            frame_indices = list(range(ep_start, ep_start + ep_len))
        else:
            start = random.randint(ep_start, ep_start + ep_len - self.n_frames)
            frame_indices = list(range(start, start + self.n_frames))

        # Collect frames
        images = []
        actions = []
        for fi in frame_indices:
            sample = self._ds[fi]
            img = sample[self._image_key]  # (C, H, W)
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            images.append(img)
            if self._has_action:
                actions.append(sample["action"])

        video = torch.stack(images, dim=0)  # (T_actual, C, H, W)

        # Pad if too short
        T_actual = video.shape[0]
        if T_actual < self.n_frames:
            pad = self.n_frames - T_actual
            video = torch.cat([video, video[-1:].expand(pad, -1, -1, -1)], dim=0)

        # Resize to target resolution
        C, H, W = video.shape[1], video.shape[2], video.shape[3]
        if H != self.resolution or W != self.resolution:
            video = F.interpolate(
                video, size=(self.resolution, self.resolution),
                mode="bilinear", align_corners=False,
            )

        # Skip ImageNet normalization — LeRobot images are already [0,1]
        # and may not be natural images (e.g. PushT is 2D renders)
        result = {"video": video}

        if actions:
            act = torch.stack(actions, dim=0)  # (T_actual, d_action)
            if act.shape[0] < self.n_frames:
                pad = self.n_frames - act.shape[0]
                act = torch.cat([act, act[-1:].expand(pad, -1)], dim=0)
            result["action"] = act

        return result


# ═══════════════════════════════════════════════════════════════════════════
# WebDataset pipeline (for large-scale sharded tars)
# ═══════════════════════════════════════════════════════════════════════════

def _decode_webdataset_sample(
    sample: dict[str, Any],
    n_frames: int,
    resolution: int,
    transform: Callable | None,
) -> dict[str, torch.Tensor]:
    """Decode a single webdataset sample containing a video file."""
    # WebDataset decodes video bytes — find the video key
    video_data = None
    for key in sample:
        if any(key.endswith(ext) for ext in (".mp4", ".avi", ".mov", ".mkv")):
            video_data = sample[key]
            break

    if video_data is None:
        raise KeyError(f"No video key found in sample, keys: {list(sample.keys())}")

    if not DECORD_AVAILABLE:
        return SyntheticVideoDataset(n_samples=1, n_frames=n_frames, resolution=resolution)[0]

    # Decode from bytes
    vr = decord.VideoReader(decord.bridge.io.BytesIO(video_data), num_threads=1)
    total_frames = len(vr)
    if total_frames >= n_frames:
        indices = torch.linspace(0, total_frames - 1, n_frames).long().tolist()
    else:
        indices = list(range(total_frames))

    frames = vr.get_batch(indices)
    video = frames.permute(0, 3, 1, 2).float() / 255.0

    if transform is not None:
        video = transform(video)

    return {"video": video}


def build_webdataset(
    shard_urls: str,
    n_frames: int = 16,
    resolution: int = 224,
    batch_size: int = 16,
    num_workers: int = 4,
    transform: Callable | None = None,
) -> DataLoader:
    """Build a WebDataset pipeline from tar shard URLs.

    Args:
        shard_urls: Brace-expanded URL pattern, e.g. "data/shards-{000..099}.tar".
        n_frames: Frames per clip.
        resolution: Spatial resolution.
        batch_size: Batch size per worker.
        num_workers: DataLoader workers.
        transform: Video transforms (defaults to training augmentations).

    Returns:
        DataLoader yielding dicts with 'video' tensors.
    """
    if not WEBDATASET_AVAILABLE:
        raise ImportError(
            "webdataset is required for shard-based data loading. "
            "Install with: pip install webdataset"
        )

    if transform is None:
        transform = build_train_transforms(n_frames, resolution)

    dataset = (
        wds.WebDataset(shard_urls, shardshuffle=True)
        .shuffle(1000)
        .map(lambda sample: _decode_webdataset_sample(sample, n_frames, resolution, transform))
    )

    loader = wds.WebLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    return loader


# ═══════════════════════════════════════════════════════════════════════════
# Factory: build_dataloader()
# ═══════════════════════════════════════════════════════════════════════════

def build_dataloader(
    config: Any,
    split: str = "train",
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
) -> DataLoader:
    """Build a DataLoader from TrainingConfig.

    Dispatches to SyntheticVideoDataset, VideoDataset, or WebDataset
    based on ``config.training.dataset_name``.

    Args:
        config: HCLSMConfig instance.
        split: "train" or "val".
        distributed: Whether to use DistributedSampler.
        world_size: Total number of processes.
        rank: Current process rank.

    Returns:
        DataLoader ready for iteration.
    """
    tc = config.training
    pc = config.perception
    n_frames = pc.temporal_resolution
    resolution = pc.input_resolution

    is_train = split == "train"
    transform = build_train_transforms(n_frames, resolution) if is_train else build_eval_transforms(n_frames, resolution)

    # ── Synthetic ──
    if tc.dataset_name == "synthetic":
        dataset = SyntheticVideoDataset(
            n_samples=max(tc.batch_size * 10, 1000),
            n_frames=n_frames,
            resolution=resolution,
        )

    # ── WebDataset (sharded tars) ──
    elif tc.dataset_name == "webdataset":
        shard_pattern = os.path.join(tc.data_dir, split, "*.tar")
        return build_webdataset(
            shard_urls=shard_pattern,
            n_frames=n_frames,
            resolution=resolution,
            batch_size=tc.batch_size,
            num_workers=tc.num_workers,
            transform=transform,
        )

    # ── OpenX-Embodiment (LeRobot) ──
    elif tc.dataset_name.startswith("openx") or tc.dataset_name.startswith("lerobot"):
        repo_id = tc.data_dir if tc.data_dir else "lerobot/pusht"
        dataset = OpenXDataset(
            repo_id=repo_id,
            n_frames=n_frames,
            resolution=resolution,
        )

    # ── Video files from directory ──
    else:
        data_dir = os.path.join(tc.data_dir, split) if tc.data_dir else tc.data_dir
        if not data_dir or not os.path.isdir(data_dir):
            logger.warning(
                f"data_dir '{data_dir}' not found, falling back to synthetic dataset"
            )
            dataset = SyntheticVideoDataset(
                n_samples=max(tc.batch_size * 10, 1000),
                n_frames=n_frames,
                resolution=resolution,
            )
        else:
            dataset = VideoDataset.from_directory(
                data_dir, n_frames=n_frames, resolution=resolution, transform=transform,
            )

    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=is_train,
        )

    return DataLoader(
        dataset,
        batch_size=tc.batch_size,
        shuffle=(is_train and sampler is None),
        num_workers=tc.num_workers,
        pin_memory=True,
        drop_last=is_train,
        sampler=sampler,
    )
