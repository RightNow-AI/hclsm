"""Object state representation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ObjectState:
    """Structured representation of objects in the world state.

    Each field has shape (B, T, N_max, ...) where N_max is the maximum
    number of object slots. The alive mask indicates which slots are active.
    """

    z: torch.Tensor  # (B, T, N_max, d_slot) — latent state vector
    p: torch.Tensor  # (B, T, N_max, 3) — 3D position (learned)
    v: torch.Tensor  # (B, T, N_max, 3) — velocity
    a: torch.Tensor  # (B, T, N_max, d_attr) — attributes
    h: torch.Tensor  # (B, T, N_max, d_hidden) — hidden state for dynamics
    alive: torch.Tensor  # (B, T, N_max) — existence probability in [0, 1]

    @property
    def alive_hard(self) -> torch.Tensor:
        """Binary alive mask via straight-through estimator."""
        return (self.alive > 0.5).float() - self.alive.detach() + self.alive

    @property
    def n_alive(self) -> torch.Tensor:
        """Number of alive slots per sample per timestep. Shape (B, T)."""
        return (self.alive > 0.5).float().sum(dim=-1)

    def masked(self, field_name: str) -> torch.Tensor:
        """Get a field with dead slots zeroed out."""
        val = getattr(self, field_name)
        mask = self.alive_hard
        while mask.dim() < val.dim():
            mask = mask.unsqueeze(-1)
        return val * mask
