"""Synthetic causal environments for validating causal discovery.

CausalBlockWorld: Simple 2D environment with objects that have KNOWN
causal relationships (push, gravity, collision). Used to verify that
the learned causal graph recovers the ground-truth causal structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class CausalSceneConfig:
    """Configuration for a causal scene."""

    n_objects: int = 5
    n_frames: int = 16
    resolution: int = 64
    d_slot: int = 8
    max_causal_edges: int = 6


class CausalBlockWorld(Dataset):
    """Synthetic causal block world with known causal structure.

    Generates scenes with colored blocks that interact causally:
    - Block A pushes Block B (A→B edge in causal graph)
    - Gravity pulls blocks down
    - Collisions transfer momentum

    The ground-truth causal graph is generated alongside the video,
    enabling direct evaluation of causal discovery accuracy.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        config: CausalSceneConfig | None = None,
    ) -> None:
        self.n_samples = n_samples
        self.config = config or CausalSceneConfig()

    def __len__(self) -> int:
        return self.n_samples

    def _generate_causal_graph(
        self, n_obj: int, device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Generate a random DAG over n_obj objects.

        Returns:
            adj: (n_obj, n_obj) binary adjacency matrix (DAG).
        """
        # Generate random lower-triangular matrix (guaranteed DAG)
        adj = torch.zeros(n_obj, n_obj, device=device)
        n_edges = min(
            torch.randint(1, self.config.max_causal_edges + 1, (1,)).item(),
            n_obj * (n_obj - 1) // 2,
        )

        # Sample edges from lower triangle
        possible_edges = []
        for i in range(n_obj):
            for j in range(i + 1, n_obj):
                possible_edges.append((i, j))

        if possible_edges and n_edges > 0:
            indices = torch.randperm(len(possible_edges))[:n_edges]
            for idx in indices:
                i, j = possible_edges[idx]
                adj[i, j] = 1.0  # i causes j

        return adj

    def _simulate(
        self, n_obj: int, causal_graph: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Simulate object trajectories governed by the causal graph.

        Objects have positions and velocities. Causal edges mean:
        "when object i moves, object j is pushed in the same direction."

        Returns:
            states: (T, n_obj, d_slot) object state trajectories.
            video: (T, 3, H, W) rendered video frames.
        """
        T = self.config.n_frames
        H = W = self.config.resolution
        d_slot = self.config.d_slot

        # Initialize objects: position (2D), velocity (2D), color (3), size (1)
        positions = torch.rand(n_obj, 2) * 0.8 + 0.1  # [0.1, 0.9]
        velocities = torch.randn(n_obj, 2) * 0.02
        colors = torch.rand(n_obj, 3)
        sizes = torch.rand(n_obj, 1) * 10 + 5  # [5, 15] pixels

        states = torch.zeros(T, n_obj, d_slot)
        video = torch.zeros(T, 3, H, W)

        for t in range(T):
            # Apply causal effects: if i→j, j's velocity gets a push from i
            causal_push = causal_graph.T @ velocities  # (n_obj, 2)
            velocities = velocities + causal_push * 0.3

            # Apply gravity (downward)
            velocities[:, 1] = velocities[:, 1] + 0.005

            # Update positions
            positions = positions + velocities

            # Bounce off walls
            for dim in range(2):
                below = positions[:, dim] < 0
                above = positions[:, dim] > 1
                velocities[below, dim] = velocities[below, dim].abs()
                velocities[above, dim] = -velocities[above, dim].abs()
                positions[:, dim] = positions[:, dim].clamp(0, 1)

            # Damping
            velocities = velocities * 0.98

            # Store state: [pos_x, pos_y, vel_x, vel_y, color_r, color_g, color_b, size]
            states[t, :, 0:2] = positions
            states[t, :, 2:4] = velocities
            states[t, :, 4:7] = colors
            states[t, :, 7:8] = sizes / 15.0  # Normalize

            # Render to video
            for obj_i in range(n_obj):
                px = int(positions[obj_i, 0] * (W - 1))
                py = int(positions[obj_i, 1] * (H - 1))
                sz = int(sizes[obj_i].item())
                x0 = max(0, px - sz // 2)
                x1 = min(W, px + sz // 2 + 1)
                y0 = max(0, py - sz // 2)
                y1 = min(H, py + sz // 2 + 1)
                video[t, :, y0:y1, x0:x1] = colors[obj_i, :, None, None]

        return states, video

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Generate a causal block world sample.

        Returns:
            Dict with:
                'video': (T, 3, H, W) rendered video.
                'states': (T, n_obj, d_slot) ground-truth object states.
                'causal_graph': (n_obj, n_obj) ground-truth causal adjacency.
                'n_objects': scalar tensor.
        """
        n_obj = self.config.n_objects
        causal_graph = self._generate_causal_graph(n_obj)
        states, video = self._simulate(n_obj, causal_graph)

        return {
            "video": video,
            "states": states,
            "causal_graph": causal_graph,
            "n_objects": torch.tensor(n_obj),
        }


def causal_discovery_accuracy(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate causal discovery accuracy.

    Args:
        predicted: (N, N) predicted adjacency in [0, 1].
        ground_truth: (N, N) ground-truth binary adjacency.
        threshold: Binarization threshold for predicted.

    Returns:
        Dict with precision, recall, F1, structural Hamming distance.
    """
    pred_binary = (predicted > threshold).float()
    gt_binary = (ground_truth > 0.5).float()

    # Remove diagonal
    mask = 1 - torch.eye(pred_binary.shape[0], device=pred_binary.device)
    pred_binary = pred_binary * mask
    gt_binary = gt_binary * mask

    tp = (pred_binary * gt_binary).sum().item()
    fp = (pred_binary * (1 - gt_binary)).sum().item()
    fn = ((1 - pred_binary) * gt_binary).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Structural Hamming Distance
    shd = (pred_binary != gt_binary).float().sum().item()

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "shd": shd,
        "n_predicted_edges": pred_binary.sum().item(),
        "n_true_edges": gt_binary.sum().item(),
    }
