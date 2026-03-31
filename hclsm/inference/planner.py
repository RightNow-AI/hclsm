"""Model-based planners using world model rollouts.

Sprint 5: CEM (Cross-Entropy Method) and MPPI (Model Predictive Path Integral)
planners that use the WorldSimulator for trajectory evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from hclsm.inference.world_simulator import WorldSimulator


@dataclass
class PlanResult:
    """Result of a planning step."""

    action: torch.Tensor  # (B, d_action) best action for current step
    action_sequence: torch.Tensor  # (B, horizon, d_action) full planned sequence
    best_cost: torch.Tensor  # (B,) cost of best trajectory
    mean_cost: float  # Average cost across all samples


class CEMPlanner:
    """Cross-Entropy Method planner using world model rollouts.

    Algorithm:
    1. Initialize Gaussian prior over action sequences: N(mu, sigma²)
    2. For each CEM iteration:
        a. Sample K action sequences from current distribution
        b. Rollout each through world simulator
        c. Evaluate trajectories with cost function
        d. Select top-E elite sequences (lowest cost)
        e. Refit Gaussian to elite sequences: mu, sigma = fit(elites)
    3. Return mean of final distribution as planned action sequence
    """

    def __init__(
        self,
        simulator: WorldSimulator,
        horizon: int = 10,
        n_samples: int = 256,
        n_elites: int = 32,
        n_iterations: int = 5,
        d_action: int = 4,
        action_low: float = -1.0,
        action_high: float = 1.0,
    ) -> None:
        self.simulator = simulator
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_elites = n_elites
        self.n_iterations = n_iterations
        self.d_action = d_action
        self.action_low = action_low
        self.action_high = action_high

    @torch.no_grad()
    def plan(
        self,
        initial_states: torch.Tensor,
        alive_mask: torch.Tensor,
        goal_states: torch.Tensor | None = None,
        value_fn: nn.Module | None = None,
    ) -> PlanResult:
        """Plan an action sequence using CEM.

        Args:
            initial_states: (B, N, d_slot) current world state.
            alive_mask: (B, N) alive mask.
            goal_states: (B, N, d_slot) optional goal.
            value_fn: Optional learned value function.

        Returns:
            PlanResult with best action and full sequence.
        """
        B, N, D = initial_states.shape
        device = initial_states.device
        K = self.n_samples
        H = self.horizon
        da = self.d_action

        # Initialize distribution
        mu = torch.zeros(B, H, da, device=device)
        sigma = torch.ones(B, H, da, device=device) * 0.5

        best_cost = torch.full((B,), float("inf"), device=device)
        best_sequence = mu.clone()

        for iteration in range(self.n_iterations):
            # Sample action sequences: (B, K, H, d_action)
            noise = torch.randn(B, K, H, da, device=device)
            actions = mu.unsqueeze(1) + sigma.unsqueeze(1) * noise
            actions = actions.clamp(self.action_low, self.action_high)

            # Rollout all sequences in parallel
            rollout = self.simulator.batched_rollout(
                initial_states, alive_mask, actions, n_steps=H,
            )

            # Evaluate trajectories: (B*K,) → reshape to (B, K)
            costs = self.simulator.evaluate_trajectory(
                rollout.predicted_states,
                rollout.alive_masks,
                goal_states=goal_states.unsqueeze(1).expand(B, K, N, D).reshape(B * K, N, D) if goal_states is not None else None,
                value_fn=value_fn,
            ).reshape(B, K)

            # Select elites (lowest cost)
            elite_costs, elite_idx = costs.topk(self.n_elites, dim=1, largest=False)

            # Gather elite actions: (B, n_elites, H, da)
            elite_idx_exp = elite_idx.unsqueeze(-1).unsqueeze(-1).expand(B, self.n_elites, H, da)
            elite_actions = actions.gather(1, elite_idx_exp)

            # Refit distribution to elites
            mu = elite_actions.mean(dim=1)  # (B, H, da)
            sigma = elite_actions.std(dim=1).clamp(min=0.01)  # (B, H, da)

            # Track best
            iter_best_cost, iter_best_idx = costs.min(dim=1)
            improved = iter_best_cost < best_cost
            if improved.any():
                for b in range(B):
                    if improved[b]:
                        best_cost[b] = iter_best_cost[b]
                        best_sequence[b] = actions[b, iter_best_idx[b]]

        return PlanResult(
            action=best_sequence[:, 0],  # First action
            action_sequence=best_sequence,
            best_cost=best_cost,
            mean_cost=best_cost.mean().item(),
        )


class MPPIPlanner:
    """Model Predictive Path Integral planner.

    Algorithm:
    1. Sample K action sequences with noise around current plan
    2. Rollout each through world simulator
    3. Compute trajectory costs
    4. Soft-weight trajectories by exp(-cost / temperature)
    5. Return weighted mean as planned action sequence

    MPPI is softer than CEM — it uses all trajectories weighted by quality
    rather than hard selection of elites.
    """

    def __init__(
        self,
        simulator: WorldSimulator,
        horizon: int = 10,
        n_samples: int = 256,
        temperature: float = 1.0,
        noise_sigma: float = 0.3,
        d_action: int = 4,
        action_low: float = -1.0,
        action_high: float = 1.0,
    ) -> None:
        self.simulator = simulator
        self.horizon = horizon
        self.n_samples = n_samples
        self.temperature = temperature
        self.noise_sigma = noise_sigma
        self.d_action = d_action
        self.action_low = action_low
        self.action_high = action_high

        # Warm-start: shift previous plan forward
        self._prev_plan: torch.Tensor | None = None

    @torch.no_grad()
    def plan(
        self,
        initial_states: torch.Tensor,
        alive_mask: torch.Tensor,
        goal_states: torch.Tensor | None = None,
        value_fn: nn.Module | None = None,
    ) -> PlanResult:
        """Plan an action sequence using MPPI.

        Args:
            initial_states: (B, N, d_slot) current world state.
            alive_mask: (B, N) alive mask.
            goal_states: (B, N, d_slot) optional goal.
            value_fn: Optional learned value function.

        Returns:
            PlanResult with weighted-mean action and full sequence.
        """
        B, N, D = initial_states.shape
        device = initial_states.device
        K = self.n_samples
        H = self.horizon
        da = self.d_action

        # Warm-start from previous plan (shifted by 1 timestep)
        if self._prev_plan is not None and self._prev_plan.shape[0] == B:
            nominal = torch.zeros(B, H, da, device=device)
            nominal[:, :-1] = self._prev_plan[:, 1:]
            # Last step: repeat final action
            nominal[:, -1] = self._prev_plan[:, -1]
        else:
            nominal = torch.zeros(B, H, da, device=device)

        # Sample perturbations
        noise = torch.randn(B, K, H, da, device=device) * self.noise_sigma
        actions = nominal.unsqueeze(1) + noise  # (B, K, H, da)
        actions = actions.clamp(self.action_low, self.action_high)

        # Rollout
        rollout = self.simulator.batched_rollout(
            initial_states, alive_mask, actions, n_steps=H,
        )

        # Evaluate costs: (B*K,) → (B, K)
        goal_exp = None
        if goal_states is not None:
            goal_exp = goal_states.unsqueeze(1).expand(B, K, N, D).reshape(B * K, N, D)

        costs = self.simulator.evaluate_trajectory(
            rollout.predicted_states,
            rollout.alive_masks,
            goal_states=goal_exp,
            value_fn=value_fn,
        ).reshape(B, K)

        # MPPI weighting: w_k = exp(-cost_k / temperature)
        # Normalize per batch
        min_cost = costs.min(dim=1, keepdim=True).values
        weights = torch.exp(-(costs - min_cost) / self.temperature)  # (B, K)
        weights = weights / weights.sum(dim=1, keepdim=True)  # Normalize

        # Weighted mean action sequence
        weights_exp = weights.unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, 1)
        planned = (actions * weights_exp).sum(dim=1)  # (B, H, da)
        planned = planned.clamp(self.action_low, self.action_high)

        # Cache for warm-start
        self._prev_plan = planned.detach()

        best_cost = costs.min(dim=1).values

        return PlanResult(
            action=planned[:, 0],
            action_sequence=planned,
            best_cost=best_cost,
            mean_cost=best_cost.mean().item(),
        )

    def reset(self) -> None:
        """Reset warm-start state (call when environment resets)."""
        self._prev_plan = None
