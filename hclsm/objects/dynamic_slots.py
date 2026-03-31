"""Dynamic Slot Attention with variable-count object slots.

THE CORE INNOVATION: Unlike standard Slot Attention (fixed N slots), this module
supports learned slot birth/death with differentiable existence prediction.

CRITICAL: The attention softmax is over the SLOT dimension, NOT the token dimension.
This creates competition between slots for input tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from hclsm.config import ObjectConfig


class DynamicSlotAttention(nn.Module):
    """Variable-count slot attention with learned slot birth/death.

    Architecture:
    1. Initialize N_max slot proposals from learned Gaussian
    2. Iterative attention refinement (n_iterations rounds)
    3. Existence head predicts alive probability per slot
    4. Slot birth when residual attention energy is high

    Args:
        config: ObjectConfig with d_slot, n_max_slots, n_iterations, etc.
        d_input: Dimension of input tokens (d_world).
    """

    def __init__(
        self, config: ObjectConfig, d_input: int,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.d_slot = config.d_slot
        self.n_max_slots = config.n_max_slots
        self.n_iterations = config.n_iterations
        self.existence_threshold = config.existence_threshold
        self.birth_threshold = config.birth_threshold
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Slot initialization parameters
        self.slot_mu = nn.Parameter(torch.randn(config.d_slot) * 0.02)
        self.slot_log_sigma = nn.Parameter(torch.full((config.d_slot,), -2.0))

        # Attention projections
        self.project_k = nn.Linear(d_input, config.d_slot, bias=False)
        self.project_v = nn.Linear(d_input, config.d_slot, bias=False)
        self.project_q = nn.Linear(config.d_slot, config.d_slot, bias=False)

        # GRU for slot update — orthogonal init prevents gate saturation
        self.gru = nn.GRUCell(config.d_slot, config.d_slot)
        nn.init.orthogonal_(self.gru.weight_ih)
        nn.init.orthogonal_(self.gru.weight_hh)
        nn.init.zeros_(self.gru.bias_ih)
        nn.init.zeros_(self.gru.bias_hh)

        # MLP refinement after GRU
        self.norm_slots = nn.LayerNorm(config.d_slot)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_slot, config.d_slot * 4),
            nn.ReLU(),
            nn.Linear(config.d_slot * 4, config.d_slot),
        )

        # Normalization for attention
        self.norm_input = nn.LayerNorm(d_input)
        self.norm_pre_attn = nn.LayerNorm(config.d_slot)

        # Existence head
        self.existence_head = nn.Sequential(
            nn.Linear(config.d_slot, config.d_slot // 4),
            nn.ReLU(),
            nn.Linear(config.d_slot // 4, 1),
        )

        # Slot birth projection
        self.birth_proj = nn.Linear(d_input, config.d_slot)

        self.scale = config.d_slot ** -0.5

    def _init_slots(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize slot proposals from learned Gaussian.

        Returns:
            slots: (B, N_max, d_slot)
        """
        sigma = self.slot_log_sigma.exp()
        eps = torch.randn(
            batch_size, self.n_max_slots, self.d_slot, device=device
        )
        return self.slot_mu + sigma * eps

    def _iteration_step(
        self, slots: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    ) -> torch.Tensor:
        """Single slot attention iteration (checkpointable).

        Args:
            slots: (B, N_max, d_slot) current slot states.
            k: (B, M, d_slot) projected input keys.
            v: (B, M, d_slot) projected input values.

        Returns:
            Updated slots (B, N_max, d_slot).
        """
        B = slots.shape[0]
        slots_normed = self.norm_pre_attn(slots)
        q = self.project_q(slots_normed)

        # CRITICAL: softmax over SLOT dimension (dim=1)
        attn_logits = torch.bmm(q, k.transpose(1, 2)) * self.scale
        # Clamp logits to prevent softmax overflow
        attn_logits = torch.clamp(attn_logits, min=-30.0, max=30.0)
        attn = F.softmax(attn_logits, dim=1)

        updates = torch.bmm(attn, v)

        slots = self.gru(
            updates.reshape(-1, self.d_slot),
            slots.reshape(-1, self.d_slot),
        ).reshape(B, self.n_max_slots, self.d_slot)

        slots = slots + self.mlp(self.norm_slots(slots))
        # Prevent magnitude growth across iterations (key for base/large configs)
        slots = torch.clamp(slots, min=-50.0, max=50.0)
        return slots

    def forward(
        self,
        inputs: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract object slots from input tokens.

        Args:
            inputs: (B, M, d_input) — input tokens (e.g., patch embeddings).
            return_attention: If True, also return final attention map.

        Returns:
            slots: (B, N_max, d_slot) — object slot representations.
            alive: (B, N_max) — existence probability per slot.
            attn: (B, N_max, M) — final attention map (only if return_attention=True).
        """
        B, M, _ = inputs.shape
        device = inputs.device

        inputs_norm = self.norm_input(inputs)
        k = self.project_k(inputs_norm)  # (B, M, d_slot)
        v = self.project_v(inputs_norm)  # (B, M, d_slot)

        slots = self._init_slots(B, device)  # (B, N_max, d_slot)

        for _ in range(self.n_iterations):
            if self.use_gradient_checkpointing and self.training:
                slots = torch.utils.checkpoint.checkpoint(
                    self._iteration_step, slots, k, v, use_reentrant=False,
                )
            else:
                slots = self._iteration_step(slots, k, v)

        # Existence prediction
        alive = self.existence_head(slots).squeeze(-1)  # (B, N_max)
        alive = torch.sigmoid(alive)

        # Compute final attention for visualization
        final_attn = None
        if return_attention:
            with torch.no_grad():
                slots_normed = self.norm_pre_attn(slots)
                q = self.project_q(slots_normed)
                attn_logits = torch.bmm(q, k.transpose(1, 2)) * self.scale
                final_attn = F.softmax(attn_logits, dim=1)  # (B, N_max, M)

        # Slot birth: check residual attention energy
        with torch.no_grad():
            slots_normed = self.norm_pre_attn(slots)
            q = self.project_q(slots_normed)
            attn_logits = torch.bmm(q, k.transpose(1, 2)) * self.scale
            attn = F.softmax(attn_logits, dim=1)

            attn_received = attn.sum(dim=1)  # (B, M)
            residual = (1.0 - attn_received).abs()
            max_residual = residual.max(dim=-1).values  # (B,)

            dormant = alive < self.existence_threshold
            needs_birth = max_residual > self.birth_threshold

            if needs_birth.any():
                max_token_idx = residual.argmax(dim=-1)
                for b in range(B):
                    if needs_birth[b] and dormant[b].any():
                        first_dormant = dormant[b].nonzero(as_tuple=True)[0][0]
                        token_feat = inputs[b, max_token_idx[b]]
                        slots[b, first_dormant] = self.birth_proj(token_feat)

        if return_attention:
            return slots, alive, final_attn
        return slots, alive
