"""Level 2 goal dynamics — compressed Transformer for planning.

Sprint 5: Language/goal conditioning via cross-attention, goal-directed
abstract state prediction, and state decompression for top-down flow.

Operates on abstract state representations compressed from Level 1 events.
Handles goals, subgoals, and multi-step reasoning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from hclsm.config import Level2Config


class Level2GoalDynamics(nn.Module):
    """Compressed Transformer for high-level goal/plan dynamics.

    Architecture:
    1. Pool event-level object states → event sequence
    2. Cross-attention: learned summary queries attend to events
    3. Optional: goal tokens cross-attend with summaries
    4. Transformer layers for goal-level prediction
    5. Output: abstract states for top-down conditioning

    Goal conditioning is optional — when d_goal=0, behaves as Sprint 1.
    """

    def __init__(self, config: Level2Config, d_slot: int) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.d_slot = d_slot
        self.n_summary_tokens = config.n_summary_tokens
        self.min_events = 4
        self.d_goal = config.d_goal

        # Learned summary query tokens
        self.summary_queries = nn.Parameter(
            torch.randn(config.n_summary_tokens, config.d_model) * 0.02
        )

        # Project event states to d_model
        self.proj_events = nn.Linear(d_slot, config.d_model, bias=False)

        # Cross-attention for compression (events → summaries)
        self.compress_attn = nn.MultiheadAttention(
            config.d_model, config.n_heads, batch_first=True, bias=False,
        )
        self.compress_norm_q = nn.LayerNorm(config.d_model)
        self.compress_norm_kv = nn.LayerNorm(config.d_model)

        # Goal conditioning (optional)
        if config.d_goal > 0:
            self.goal_proj = nn.Sequential(
                nn.Linear(config.d_goal, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, config.d_model),
            )
            self.goal_cross_attn = nn.MultiheadAttention(
                config.d_model, config.n_heads, batch_first=True, bias=False,
            )
            self.goal_norm_q = nn.LayerNorm(config.d_model)
            self.goal_norm_kv = nn.LayerNorm(config.d_model)

        # Transformer for goal-level processing
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.d_model * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(config.n_layers)
        ])

        self.norm = nn.LayerNorm(config.d_model)

        # State decompression: project back to d_slot for top-down conditioning
        self.decompress = nn.Linear(config.d_model, d_slot, bias=False)

    def forward(
        self,
        event_states: torch.Tensor,
        event_pad_mask: torch.Tensor | None = None,
        goal: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Compress events into abstract goal-level states.

        Args:
            event_states: (B, K, N_max, d_slot) event-level object states.
            event_pad_mask: (B, K) True for real events.
            goal: (B, d_goal) optional goal/language embedding.

        Returns:
            abstract_states: (B, n_summary_tokens, d_model) or None if too few events.
        """
        B, K, N, D = event_states.shape

        if K < self.min_events:
            return None

        # Pool over objects per event: (B, K, d_slot)
        event_pooled = event_states.mean(dim=2)

        # Project to d_model
        event_proj = self.proj_events(event_pooled)  # (B, K, d_model)

        # Cross-attention: learned queries attend to events
        queries = self.summary_queries.unsqueeze(0).expand(B, -1, -1)
        queries = self.compress_norm_q(queries)
        kv = self.compress_norm_kv(event_proj)

        kpm = None
        if event_pad_mask is not None:
            kpm = ~event_pad_mask.bool()

        abstract, _ = self.compress_attn(
            queries, kv, kv, key_padding_mask=kpm,
        )  # (B, n_summary, d_model)

        # Goal conditioning via cross-attention
        if goal is not None and self.d_goal > 0:
            goal_tokens = self.goal_proj(goal).unsqueeze(1)  # (B, 1, d_model)
            q = self.goal_norm_q(abstract)
            kv_goal = self.goal_norm_kv(goal_tokens)
            goal_context, _ = self.goal_cross_attn(q, kv_goal, kv_goal)
            abstract = abstract + goal_context

        # Transformer processing
        for block in self.blocks:
            abstract = block(abstract)

        abstract = self.norm(abstract)
        return abstract

    def decompress_to_slot_dim(self, abstract: torch.Tensor) -> torch.Tensor:
        """Project abstract states back to d_slot for top-down conditioning.

        Args:
            abstract: (B, n_summary, d_model)

        Returns:
            (B, d_slot) pooled and projected context.
        """
        pooled = abstract.mean(dim=1)  # (B, d_model)
        return self.decompress(pooled)  # (B, d_slot)
