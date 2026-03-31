"""World simulation engine — rolls the model forward in time.

Sprint 5: Full autoregressive rollout with action conditioning,
batched rollout for planning, and trajectory output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from hclsm.causality.action_conditioner import ActionConditioner


@dataclass
class RolloutOutput:
    """Output of a world model rollout."""

    predicted_states: torch.Tensor  # (B, T_rollout, N, d_slot)
    alive_masks: torch.Tensor  # (B, T_rollout, N)
    event_masks: torch.Tensor | None = None  # (B, T_rollout)
    abstract_states: list[torch.Tensor | None] = field(default_factory=list)


class WorldSimulator:
    """Rolls the HCLSM world model forward for prediction/planning.

    Supports:
    - Open-loop rollout (no actions)
    - Action-conditioned rollout (inject actions at each step)
    - Batched rollout (multiple action sequences in parallel)
    - SSM-based dynamics for rollout (uses model's Level0SSM when available)
    """

    def __init__(
        self,
        model: nn.Module,
        action_conditioner: ActionConditioner | None = None,
    ) -> None:
        self.model = model
        self.action_conditioner = action_conditioner

        # Extract SSM dynamics from model if available
        self._ssm = getattr(model, "level0_ssm", None)

    @torch.no_grad()
    def encode(
        self, video_frames: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode video frames into object slots.

        Args:
            video_frames: (B, T, C, H, W)

        Returns:
            slots: (B, T, N, d_slot)
            alive: (B, T, N)
        """
        output = self.model(video_frames)
        return output.predicted_states, output.alive_mask

    @torch.no_grad()
    def rollout(
        self,
        initial_frames: torch.Tensor,
        n_steps: int = 10,
        actions: torch.Tensor | None = None,
        goal: torch.Tensor | None = None,
    ) -> RolloutOutput:
        """Roll forward n_steps from initial observation.

        Args:
            initial_frames: (B, T_context, C, H, W) context frames.
            n_steps: Number of future steps to predict.
            actions: (B, n_steps, d_action) optional actions per step.
            goal: (B, d_goal) optional goal for Level 2 conditioning.

        Returns:
            RolloutOutput with predicted trajectories.
        """
        # Encode initial context
        output = self.model(initial_frames)
        current_states = output.predicted_states  # (B, T, N, d_slot)
        alive = output.alive_mask  # (B, T, N)

        B, T_ctx, N, D = current_states.shape

        # Use last frame's states as starting point
        last_states = current_states[:, -1]  # (B, N, d_slot)
        last_alive = alive[:, -1]  # (B, N)

        predicted_states = []
        alive_masks = []

        # Try SSM-based rollout first, fall back to momentum
        use_ssm = self._ssm is not None

        if use_ssm and n_steps >= 2:
            # Collect context + future into SSM input: (B, N, T_future, d)
            # Start from the last context states repeated for n_steps
            ssm_input = last_states.unsqueeze(2).expand(B, N, n_steps, D)
            obj_mask = (last_alive > 0.5).float()

            # Apply actions to the SSM input if available
            if actions is not None and self.action_conditioner is not None:
                frames_list = []
                s = last_states
                for t in range(n_steps):
                    s = self.action_conditioner(s, actions[:, t])
                    frames_list.append(s)
                ssm_input = torch.stack(frames_list, dim=2)  # (B, N, T, d)

            ssm_out = self._ssm(ssm_input, obj_mask)  # (B, N, T, d)
            ssm_out = ssm_out.permute(0, 2, 1, 3)  # (B, T, N, d)

            for t in range(n_steps):
                predicted_states.append(ssm_out[:, t])
                alive_masks.append(last_alive)
        else:
            # Fallback: momentum-based prediction
            for t in range(n_steps):
                if actions is not None and self.action_conditioner is not None:
                    last_states = self.action_conditioner(last_states, actions[:, t])

                if t > 0 and len(predicted_states) >= 2:
                    velocity = predicted_states[-1] - predicted_states[-2]
                    last_states = last_states + velocity * 0.9

                predicted_states.append(last_states)
                alive_masks.append(last_alive)

        return RolloutOutput(
            predicted_states=torch.stack(predicted_states, dim=1),
            alive_masks=torch.stack(alive_masks, dim=1),
        )

    @torch.no_grad()
    def batched_rollout(
        self,
        initial_states: torch.Tensor,
        alive_mask: torch.Tensor,
        action_sequences: torch.Tensor,
        n_steps: int | None = None,
    ) -> RolloutOutput:
        """Rollout multiple action sequences in parallel from same initial state.

        Used by planners (CEM, MPPI) to evaluate many candidate action sequences.

        Args:
            initial_states: (B, N, d_slot) initial object states (already encoded).
            alive_mask: (B, N) alive mask.
            action_sequences: (B, K, T, d_action) K action sequences per batch.
            n_steps: Override for rollout length (default: action_sequences.shape[2]).

        Returns:
            RolloutOutput with shape (B*K, T, N, d_slot).
        """
        B, K, T_act, d_action = action_sequences.shape
        N, D = initial_states.shape[1], initial_states.shape[2]

        if n_steps is None:
            n_steps = T_act

        # Expand initial states: (B, N, D) → (B*K, N, D)
        states = initial_states.unsqueeze(1).expand(B, K, N, D).reshape(B * K, N, D)
        alive = alive_mask.unsqueeze(1).expand(B, K, N).reshape(B * K, N)
        actions_flat = action_sequences.reshape(B * K, T_act, d_action)

        predicted = []
        alive_list = []

        use_ssm = self._ssm is not None

        if use_ssm and n_steps >= 2:
            obj_mask = (alive > 0.5).float()
            if self.action_conditioner is not None:
                frames_list = []
                s = states
                for t in range(n_steps):
                    if t < T_act:
                        s = self.action_conditioner(s, actions_flat[:, t])
                    frames_list.append(s)
                ssm_input = torch.stack(frames_list, dim=2)  # (B*K, N, T, d)
            else:
                ssm_input = states.unsqueeze(2).expand(B * K, N, n_steps, D)

            ssm_out = self._ssm(ssm_input, obj_mask).permute(0, 2, 1, 3)
            for t in range(n_steps):
                predicted.append(ssm_out[:, t])
                alive_list.append(alive)
        else:
            for t in range(n_steps):
                if self.action_conditioner is not None and t < T_act:
                    states = self.action_conditioner(states, actions_flat[:, t])
                if t > 0 and len(predicted) >= 2:
                    velocity = predicted[-1] - predicted[-2]
                    states = states + velocity * 0.9
                predicted.append(states)
                alive_list.append(alive)

        return RolloutOutput(
            predicted_states=torch.stack(predicted, dim=1),
            alive_masks=torch.stack(alive_list, dim=1),
        )

    def evaluate_trajectory(
        self,
        trajectory: torch.Tensor,
        alive_mask: torch.Tensor,
        goal_states: torch.Tensor | None = None,
        value_fn: nn.Module | None = None,
    ) -> torch.Tensor:
        """Evaluate a trajectory by computing total cost/reward.

        Args:
            trajectory: (B, T, N, d_slot) predicted states.
            alive_mask: (B, T, N).
            goal_states: (B, N, d_slot) optional goal state.
            value_fn: Optional learned value function.

        Returns:
            costs: (B,) scalar cost per trajectory (lower = better).
        """
        B, T, N, D = trajectory.shape

        if value_fn is not None:
            # Use learned value function on final state
            final_states = trajectory[:, -1]  # (B, N, D)
            final_alive = alive_mask[:, -1]  # (B, N)
            values = value_fn(final_states, final_alive)  # (B, 1)
            return -values.squeeze(-1)  # Negate: higher value = lower cost

        if goal_states is not None:
            # Goal distance: L2 to goal at each timestep
            goal_exp = goal_states.unsqueeze(1).expand_as(trajectory)
            diff = (trajectory - goal_exp).norm(dim=-1)  # (B, T, N)
            mask = alive_mask.unsqueeze(-1).expand_as(trajectory)[..., 0]
            cost_per_step = (diff * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
            # Weight final steps more
            weights = torch.linspace(0.5, 1.0, T, device=trajectory.device)
            return (cost_per_step * weights).sum(dim=-1)  # (B,)

        # Default: prefer smooth trajectories (minimize acceleration)
        if T >= 3:
            accel = trajectory[:, 2:] - 2 * trajectory[:, 1:-1] + trajectory[:, :-2]
            return accel.norm(dim=-1).mean(dim=(1, 2))  # (B,)

        return torch.zeros(B, device=trajectory.device)
