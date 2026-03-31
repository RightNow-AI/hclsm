"""Episodic memory with modern Hopfield network retrieval.

Sprint 6: Enhanced with exponential attention scaling (beta parameter),
novelty-weighted write priority, and memory-augmented state conditioning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EpisodicMemory(nn.Module):
    """Fast episodic memory for recent experiences.

    Uses a fixed-size memory bank with novelty-weighted FIFO write and
    modern Hopfield network attention for retrieval.

    Modern Hopfield retrieval (Ramsauer et al., 2021):
        attn = softmax(beta * query @ memory^T)
        retrieved = attn @ memory

    Where beta controls the sharpness of retrieval (higher = more precise,
    lower = more blended). As beta → ∞, retrieves single nearest memory.
    """

    def __init__(
        self,
        capacity: int,
        d_memory: int,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.capacity = capacity
        self.d_memory = d_memory
        self.beta = beta

        # Memory bank (not a parameter — managed externally)
        self.register_buffer("memory", torch.zeros(capacity, d_memory))
        self.register_buffer("novelty_scores", torch.zeros(capacity))
        self.register_buffer("write_idx", torch.tensor(0, dtype=torch.long))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

        # Learned projections for read
        self.query_proj = nn.Linear(d_memory, d_memory, bias=False)
        self.key_proj = nn.Linear(d_memory, d_memory, bias=False)
        self.value_proj = nn.Linear(d_memory, d_memory, bias=False)

        # Compression MLP for writing
        self.compress = nn.Linear(d_memory, d_memory)

        # Novelty estimator: predict how surprising a state is
        self.novelty_head = nn.Sequential(
            nn.Linear(d_memory, d_memory // 4),
            nn.ReLU(),
            nn.Linear(d_memory // 4, 1),
            nn.Sigmoid(),
        )

    def compute_novelty(self, states: torch.Tensor) -> torch.Tensor:
        """Compute novelty score for incoming states.

        Novelty = 1 - max_similarity to existing memories.
        Novel states are far from anything already stored.

        Args:
            states: (B, d_memory)

        Returns:
            novelty: (B,) in [0, 1], higher = more novel.
        """
        n_stored = int(self.count.item())
        if n_stored == 0:
            return torch.ones(states.shape[0], device=states.device)

        mem = self.memory[:n_stored]  # (K, d)
        # Cosine similarity to all stored memories
        states_norm = F.normalize(states, dim=-1)
        mem_norm = F.normalize(mem, dim=-1)
        sim = states_norm @ mem_norm.t()  # (B, K)
        max_sim = sim.max(dim=-1).values  # (B,)
        return 1.0 - max_sim.clamp(0, 1)

    def write(
        self,
        states: torch.Tensor,
        novelty: torch.Tensor | None = None,
    ) -> None:
        """Write compressed states to memory.

        Uses FIFO with novelty-aware replacement: when memory is full,
        overwrites the least-novel stored memory rather than strictly FIFO.

        Args:
            states: (B, d_memory) states to store.
            novelty: (B,) optional pre-computed novelty scores.
        """
        B = states.shape[0]
        compressed = self.compress(states).detach()

        if novelty is None:
            novelty = self.compute_novelty(states).detach()

        for i in range(B):
            n_stored = int(self.count.item())

            if n_stored < self.capacity:
                # Memory not full — append
                idx = n_stored
                self.count = self.count + 1
            else:
                # Memory full — replace least novel stored memory
                idx = int(self.novelty_scores[:n_stored].argmin().item())

            self.memory[idx] = compressed[i]
            self.novelty_scores[idx] = novelty[i]
            self.write_idx = torch.tensor((idx + 1) % self.capacity, dtype=torch.long)

    def read(
        self,
        query: torch.Tensor,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Retrieve from memory via modern Hopfield attention.

        Args:
            query: (B, d_memory) query vector.
            top_k: If set, only attend to top-K most similar memories.

        Returns:
            retrieved: (B, d_memory) attention-weighted memory.
        """
        n_stored = int(self.count.item())
        if n_stored == 0:
            return torch.zeros_like(query)

        mem = self.memory[:n_stored]  # (K, d)

        # Learned projections
        q = self.query_proj(query)  # (B, d)
        k = self.key_proj(mem)  # (K, d)
        v = self.value_proj(mem)  # (K, d)

        # Modern Hopfield: exponential attention with beta scaling
        scores = q @ k.t() * self.beta  # (B, K)

        if top_k is not None and top_k < n_stored:
            # Sparse retrieval: only attend to top-K
            topk_vals, topk_idx = scores.topk(top_k, dim=-1)
            mask = torch.full_like(scores, float("-inf"))
            mask.scatter_(1, topk_idx, topk_vals)
            scores = mask

        attn = F.softmax(scores, dim=-1)  # (B, K)
        return attn @ v  # (B, d)

    def reset(self) -> None:
        """Clear all stored memories."""
        self.memory.zero_()
        self.novelty_scores.zero_()
        self.write_idx.zero_()
        self.count.zero_()

    def get_statistics(self) -> dict[str, float]:
        """Get memory statistics."""
        n = int(self.count.item())
        return {
            "n_stored": n,
            "capacity": self.capacity,
            "utilization": n / self.capacity,
            "mean_novelty": self.novelty_scores[:n].mean().item() if n > 0 else 0.0,
        }
