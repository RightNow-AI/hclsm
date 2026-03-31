"""ViT-based vision encoder with SwiGLU, RoPE, and Pre-LN."""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from hclsm.config import PerceptionConfig
from hclsm.perception.tokenizer import PatchEmbedding

logger = logging.getLogger(__name__)

try:
    from flash_attn import flash_attn_func  # noqa: F401

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    logger.info("flash-attn not available; using PyTorch SDPA fallback")


def _compute_swiglu_hidden(d_model: int) -> int:
    """Compute SwiGLU hidden dimension equivalent to 4x GELU FFN."""
    raw = int(8 / 3 * d_model)
    return ((raw + 63) // 64) * 64


class RotaryPositionEncoding(nn.Module):
    """2D Rotary Position Embedding for spatial dimensions.

    Splits head_dim in half: first half for height frequencies,
    second half for width frequencies.
    """

    def __init__(self, head_dim: int, n_patches_h: int, n_patches_w: int) -> None:
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
        dim_quarter = head_dim // 4

        freqs_h = 1.0 / (10000.0 ** (torch.arange(0, dim_quarter, dtype=torch.float32) / dim_quarter))
        freqs_w = 1.0 / (10000.0 ** (torch.arange(0, dim_quarter, dtype=torch.float32) / dim_quarter))

        grid_h = torch.arange(n_patches_h, dtype=torch.float32)
        grid_w = torch.arange(n_patches_w, dtype=torch.float32)

        # (n_patches_h, dim_quarter)
        angles_h = torch.outer(grid_h, freqs_h)
        # (n_patches_w, dim_quarter)
        angles_w = torch.outer(grid_w, freqs_w)

        # Broadcast to full grid: (n_patches_h, n_patches_w, dim_quarter)
        angles_h = angles_h[:, None, :].expand(-1, n_patches_w, -1).contiguous()
        angles_w = angles_w[None, :, :].expand(n_patches_h, -1, -1).contiguous()

        # Flatten spatial dims: (n_patches, dim_quarter)
        angles_h = angles_h.reshape(-1, dim_quarter)
        angles_w = angles_w.reshape(-1, dim_quarter)

        # Full angles: (n_patches, head_dim // 2) — repeat each for sin/cos pairs
        angles = torch.cat([angles_h, angles_w], dim=-1)  # (n_patches, head_dim/2)

        self.register_buffer("cos_cached", angles.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin_cached", angles.sin()[None, :, None, :], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding.

        Args:
            x: (B, n_patches, n_heads, head_dim)

        Returns:
            x with rotary embedding applied.
        """
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        cos = self.cos_cached[:, : x.shape[1]]
        sin = self.sin_cached[:, : x.shape[1]]
        out1 = x1 * cos - x2 * sin
        out2 = x2 * cos + x1 * sin
        return torch.cat([out1, out2], dim=-1)


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network.

    FFN(x) = (SiLU(xW1) * xV) @ W2
    """

    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = _compute_swiglu_hidden(d_model)
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)  # gate
        self.w2 = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional RoPE and FlashAttention."""

    def __init__(
        self, d_model: int, n_heads: int, dropout: float = 0.0,
        rope: RotaryPositionEncoding | None = None,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.dropout_p = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = rope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attention.

        Args:
            x: (B, N, d_model)

        Returns:
            (B, N, d_model)
        """
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, N, n_heads, head_dim)

        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        # (B, n_heads, N, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        dp = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dp)

        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block with SwiGLU FFN."""

    def __init__(
        self, d_model: int, n_heads: int, dropout: float = 0.0,
        rope: RotaryPositionEncoding | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, rope)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """ViT-based vision encoder for video input.

    Architecture:
    - Patchify: 16x16 patches with Conv2d projection
    - RoPE for 2D spatial positional encoding
    - Learned absolute temporal position embeddings
    - Pre-LN Transformer blocks with SwiGLU FFN
    - FlashAttention-2 if available, else PyTorch SDPA

    Input: (B, T, C, H, W) video frames
    Output: (B, T, N_patches, d_model) patch embeddings
    """

    def __init__(
        self, config: PerceptionConfig,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.patch_embed = PatchEmbedding(config)

        n_patches_h = config.input_resolution // config.patch_size
        n_patches_w = config.input_resolution // config.patch_size
        head_dim = config.d_model // config.n_heads

        self.temporal_pos_embed = nn.Embedding(
            config.temporal_resolution, config.d_model,
        )

        # Each block gets its own RoPE instance to avoid aliasing on deepcopy
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.d_model, config.n_heads, config.dropout,
                RotaryPositionEncoding(head_dim, n_patches_h, n_patches_w),
            )
            for _ in range(config.n_layers)
        ])

        self.norm = nn.LayerNorm(config.d_model)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.temporal_pos_embed.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode video frames to patch embeddings.

        Args:
            x: Video frames (B, T, C, H, W).

        Returns:
            Patch embeddings (B, T, N_patches, d_model).
        """
        B, T = x.shape[:2]

        # Patchify: (B, T, N_patches, d_model)
        x = self.patch_embed(x)

        # Add temporal position embeddings
        t_indices = torch.arange(T, device=x.device)
        t_embed = self.temporal_pos_embed(t_indices)  # (T, d_model)
        x = x + t_embed[None, :, None, :]  # broadcast over B and N_patches

        # Process each frame independently through Transformer
        N_patches = x.shape[2]
        x = rearrange(x, "b t n d -> (b t) n d")

        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, use_reentrant=False,
                )
            else:
                x = block(x)

        x = self.norm(x)
        x = torch.clamp(x, min=-50.0, max=50.0)  # Prevent overflow from deep ViT
        x = rearrange(x, "(b t) n d -> b t n d", b=B, t=T)
        return x
