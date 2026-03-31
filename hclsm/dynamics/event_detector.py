"""Learned event boundary detector.

Sprint 4: Multi-scale temporal features (diff, magnitude, acceleration),
causal dilated convolutions, contrastive event training signal, and
minimum event rate regularization.

Monitors Level 0 SSM states and detects discontinuities that warrant
Level 1 (event-level) processing.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleTemporalFeatures(nn.Module):
    """Extract multi-scale temporal features from state sequences.

    Computes frame differences, magnitudes, and accelerations at
    multiple temporal scales (1-step, 2-step, 4-step).
    """

    def __init__(self, d_input: int) -> None:
        super().__init__()
        self.d_input = d_input

        # Project multi-scale features to a unified representation
        # 3 scales × 3 features (diff, magnitude, acceleration) = 9 feature maps
        # But we use d_input for each, so project down
        self.proj = nn.Linear(d_input * 3, d_input)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale temporal features.

        Args:
            states: (B, T, D)

        Returns:
            features: (B, T, D) multi-scale temporal features.
        """
        B, T, D = states.shape

        features = []

        for scale in [1, 2, 4]:
            if T <= scale:
                # Not enough frames for this scale
                features.append(torch.zeros(B, T, D, device=states.device))
                continue

            # Temporal difference at this scale
            diff = torch.zeros(B, T, D, device=states.device)
            diff[:, scale:] = states[:, scale:] - states[:, :-scale]

            # Magnitude of difference (per-feature L2 norm pooled over D)
            magnitude = diff.norm(dim=-1, keepdim=True).expand_as(diff)

            # Acceleration (diff of diff)
            accel = torch.zeros(B, T, D, device=states.device)
            if T > 2 * scale:
                accel[:, 2 * scale:] = diff[:, 2 * scale:] - diff[:, scale:-scale]

            features.append(diff + magnitude + accel)

        # Concatenate and project: (B, T, 3*D) → (B, T, D)
        stacked = torch.cat(features, dim=-1)  # (B, T, 3*D)
        return self.proj(stacked)


class EventDetector(nn.Module):
    """Detect event boundaries in temporal sequences.

    Sprint 4 architecture:
    1. Multi-scale temporal features (diff, magnitude, acceleration at 3 scales)
    2. Causal dilated convolutions for temporal context
    3. MLP head → scalar event probability per timestep
    4. Gumbel straight-through estimator for hard mask
    5. Contrastive event loss: events should predict state changes
    6. Minimum event rate regularization

    Designed to be cheap (~150K params) since it runs every timestep.
    """

    def __init__(
        self,
        d_input: int,
        window_size: int = 8,
        threshold: float = 0.7,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.threshold = threshold
        self.d_input = d_input

        # Multi-scale temporal features
        self.temporal_features = MultiScaleTemporalFeatures(d_input)

        # Causal dilated convolutions (look back only)
        d_hidden = d_input // 2
        self.conv1 = nn.Conv1d(d_input, d_hidden, kernel_size=3, padding=2, dilation=1)
        self.conv2 = nn.Conv1d(d_hidden, d_hidden, kernel_size=3, padding=4, dilation=2)
        self.conv3 = nn.Conv1d(d_hidden, d_hidden, kernel_size=3, padding=8, dilation=4)
        self.norm = nn.LayerNorm(d_hidden)

        # MLP head
        self.head = nn.Sequential(
            nn.Linear(d_hidden + d_input, d_input // 4),
            nn.ReLU(),
            nn.Linear(d_input // 4, 1),
        )

        # Learnable threshold (starts at specified value) — 1D tensor for FSDP compat
        self.threshold_logit = nn.Parameter(
            torch.tensor([self._inv_sigmoid(threshold)])
        )

        # Minimum event rate target (fraction of timesteps that should be events)
        self.min_event_rate = 0.1
        self.max_event_rate = 0.5

    @staticmethod
    def _inv_sigmoid(x: float) -> float:
        return math.log(x / (1 - x))

    @property
    def learned_threshold(self) -> torch.Tensor:
        return torch.sigmoid(self.threshold_logit)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Detect events in a temporal sequence.

        Args:
            states: (B, T, d_input) — mean-pooled object states over time.

        Returns:
            event_probs: (B, T) — event probability at each timestep.
            event_mask: (B, T) — hard binary mask (straight-through estimator).
        """
        B, T, D = states.shape

        # Multi-scale temporal features
        temporal_feat = self.temporal_features(states)  # (B, T, D)

        # Causal dilated convolutions
        x = temporal_feat.transpose(1, 2)  # (B, D, T)
        h1 = F.relu(self.conv1(x))[:, :, :T]   # Trim to causal (remove future padding)
        h2 = F.relu(self.conv2(h1))[:, :, :T]
        h3 = F.relu(self.conv3(h2))[:, :, :T]
        conv_out = h3.transpose(1, 2)  # (B, T, D//2)
        conv_out = self.norm(conv_out)

        # Combine conv features with raw temporal features
        combined = torch.cat([conv_out, temporal_feat], dim=-1)  # (B, T, D//2 + D)

        # MLP → scalar event probability
        event_probs = torch.sigmoid(self.head(combined).squeeze(-1))  # (B, T)

        # Hard mask with straight-through estimator
        threshold = self.learned_threshold
        event_hard = (event_probs > threshold).float()
        event_mask = event_hard - event_probs.detach() + event_probs

        # Always keep at least one event to avoid empty tensors
        if event_hard.sum(dim=1).min() == 0:
            event_mask = event_mask.clone()
            event_mask[:, 0] = 1.0

        return event_probs, event_mask

    def event_rate_loss(self, event_probs: torch.Tensor) -> torch.Tensor:
        """Regularize event rate to stay within [min_rate, max_rate].

        Penalizes both too few events (collapse) and too many events
        (everything is an event = no hierarchy).

        Args:
            event_probs: (B, T) event probabilities.

        Returns:
            Scalar loss.
        """
        rate = event_probs.mean(dim=1)  # (B,) average event rate per sequence

        # Hinge losses for min/max rate
        too_few = F.relu(self.min_event_rate - rate)  # Penalty if below min
        too_many = F.relu(rate - self.max_event_rate)  # Penalty if above max

        return (too_few + too_many).mean()

    def contrastive_event_loss(
        self,
        event_probs: torch.Tensor,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """Contrastive signal: events should coincide with large state changes.

        Events are rewarded when they align with actual discontinuities
        in the state sequence, and penalized when they fire at smooth points.

        Args:
            event_probs: (B, T) event probabilities.
            states: (B, T, D) state sequence.

        Returns:
            Scalar loss (lower = events align with state changes).
        """
        B, T, D = states.shape
        if T < 2:
            return torch.tensor(0.0, device=states.device)

        # State change magnitude at each timestep
        state_diff = (states[:, 1:] - states[:, :-1]).norm(dim=-1)  # (B, T-1)
        # Normalize to [0, 1] per sequence
        diff_max = state_diff.max(dim=1, keepdim=True).values.clamp(min=1e-8)
        state_change = state_diff / diff_max  # (B, T-1)

        # Event probs for corresponding timesteps (skip first)
        event_p = event_probs[:, 1:]  # (B, T-1)

        # Events should fire at large state changes:
        # Maximize correlation between event_prob and state_change
        # Loss = -correlation ≈ MSE(event_p, state_change)
        return F.mse_loss(event_p, state_change.detach())
