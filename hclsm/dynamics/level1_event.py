"""Level 1 event dynamics — sparse Transformer at event boundaries.

Fires ONLY at detected event timesteps. All objects attend to all objects
within each event, enabling discrete state transitions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from hclsm.config import Level1Config


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN for Transformer blocks."""

    def __init__(self, d_model: int, expansion: int = 4) -> None:
        super().__init__()
        hidden = d_model * expansion
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class EventTransformerBlock(nn.Module):
    """Pre-LN Transformer block with SwiGLU for event processing."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, bias=False,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLUFFN(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed, key_padding_mask=key_padding_mask,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class Level1EventDynamics(nn.Module):
    """Sparse Transformer that processes object states at event boundaries.

    Each event timestep is processed independently. Within each event,
    all object slots attend to each other.
    """

    def __init__(
        self, config: Level1Config, d_slot: int,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.d_slot = d_slot
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Project from slot dim to transformer dim
        self.proj_in = nn.Linear(d_slot, config.d_model, bias=False)
        self.proj_out = nn.Linear(config.d_model, d_slot, bias=False)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            EventTransformerBlock(config.d_model, config.n_heads)
            for _ in range(config.n_layers)
        ])

        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        event_states: torch.Tensor,
        event_pad_mask: torch.Tensor | None = None,
        obj_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Process object states at event timesteps.

        Args:
            event_states: (B, K, N_max, d_slot) object states at K events.
            event_pad_mask: (B, K) True for real events, False for padding.
            obj_mask: (B, N_max) True for alive objects.

        Returns:
            updated: (B, K, N_max, d_slot) updated object states.
        """
        B, K, N, D = event_states.shape

        # Flatten events into batch: (B*K, N, d_slot)
        x = rearrange(event_states, "b k n d -> (b k) n d")

        # Project to transformer dim
        x = self.proj_in(x)  # (B*K, N, d_model)

        # Attention mask for dead objects
        key_padding_mask = None
        if obj_mask is not None:
            # (B, N) -> (B*K, N): expand to all events
            key_padding_mask = (~(obj_mask > 0.5)).unsqueeze(1).expand(B, K, N)
            key_padding_mask = rearrange(key_padding_mask, "b k n -> (b k) n")

        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, key_padding_mask, use_reentrant=False,
                )
            else:
                x = block(x, key_padding_mask=key_padding_mask)

        x = self.norm(x)
        x = self.proj_out(x)  # (B*K, N, d_slot)

        x = rearrange(x, "(b k) n d -> b k n d", b=B, k=K)

        # Zero out padded events
        if event_pad_mask is not None:
            x = x * event_pad_mask[:, :, None, None]

        return x
