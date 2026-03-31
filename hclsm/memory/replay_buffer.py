"""Simple experience replay buffer."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import torch


@dataclass
class Experience:
    """A single experience for replay."""

    states: torch.Tensor  # (T, N, d_slot)
    alive: torch.Tensor  # (T, N)
    actions: torch.Tensor | None = None
    novelty: float = 0.0


class ReplayBuffer:
    """Fixed-capacity experience replay with priority sampling.

    Prioritizes high-novelty experiences (where the model was surprised).
    """

    def __init__(self, capacity: int = 10000) -> None:
        self.capacity = capacity
        self.buffer: deque[Experience] = deque(maxlen=capacity)

    def add(self, experience: Experience) -> None:
        """Add an experience to the buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> list[Experience]:
        """Sample a batch, weighted by novelty.

        Falls back to uniform sampling if all novelties are zero.
        """
        if len(self.buffer) == 0:
            return []

        n = min(batch_size, len(self.buffer))
        novelties = [e.novelty for e in self.buffer]
        total = sum(novelties)

        if total > 0:
            weights = [n / total for n in novelties]
            return random.choices(list(self.buffer), weights=weights, k=n)
        else:
            return random.sample(list(self.buffer), n)

    def __len__(self) -> int:
        return len(self.buffer)
