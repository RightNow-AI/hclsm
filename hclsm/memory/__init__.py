"""Continual learning memory for HCLSM.

Sprint 6: Episodic memory with modern Hopfield retrieval, novelty-weighted
write, semantic consolidation with EWC regularization, and replay buffer.
"""

from hclsm.memory.episodic import EpisodicMemory
from hclsm.memory.replay_buffer import ReplayBuffer, Experience
from hclsm.memory.semantic import SemanticConsolidation, ConsolidationLoop, EWCRegularizer

__all__ = [
    "EpisodicMemory",
    "ReplayBuffer",
    "Experience",
    "SemanticConsolidation",
    "ConsolidationLoop",
    "EWCRegularizer",
]
