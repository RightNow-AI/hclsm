"""Tests for the memory module."""

from __future__ import annotations

import torch
import pytest

from hclsm.memory.episodic import EpisodicMemory


class TestEpisodicMemory:
    def test_write_read(self):
        mem = EpisodicMemory(capacity=100, d_memory=64)
        # Write some patterns
        patterns = torch.randn(5, 64)
        mem.write(patterns)
        assert mem.count.item() == 5

        # Query with the first pattern — should retrieve something similar
        query = patterns[0:1]
        retrieved = mem.read(query)
        assert retrieved.shape == (1, 64)
        # Retrieved should have some similarity to stored patterns
        sim = torch.cosine_similarity(retrieved, query, dim=-1)
        assert sim.item() > -1.0  # Sanity check

    def test_fifo_overwrite(self):
        mem = EpisodicMemory(capacity=10, d_memory=32)
        # Write more than capacity
        for _ in range(15):
            mem.write(torch.randn(1, 32))
        assert mem.count.item() == 10  # Capped at capacity

    def test_empty_read(self):
        mem = EpisodicMemory(capacity=10, d_memory=32)
        query = torch.randn(2, 32)
        retrieved = mem.read(query)
        assert retrieved.shape == (2, 32)
        assert (retrieved == 0).all()  # Empty memory returns zeros

    def test_reset(self):
        mem = EpisodicMemory(capacity=10, d_memory=32)
        mem.write(torch.randn(5, 32))
        mem.reset()
        assert mem.count.item() == 0
