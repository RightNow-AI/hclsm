"""Level 0 SSM — fast continuous physics dynamics.

Per-object Mamba-style SSM for smooth trajectories, with global SSM conditioning.
Falls back to a naive sequential scan if mamba-ssm is not installed.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from hclsm.config import Level0Config

logger = logging.getLogger(__name__)

try:
    from mamba_ssm import Mamba

    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    logger.info("mamba-ssm not available; using SimplifiedSSM fallback")


class SimplifiedSSMBlock(nn.Module):
    """Naive selective SSM block in pure PyTorch.

    Sequential scan — correct but slow. Suitable for Sprint 1 with T<=16.
    """

    def __init__(self, d_model: int, d_state: int, expand: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, d_inner, bias=False)
        self.gate_proj = nn.Linear(d_model, d_inner, bias=False)

        # Input-dependent SSM parameters
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)
        self.B_proj = nn.Linear(d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(d_inner, d_state, bias=False)

        # Learnable log-scale SSM A parameter — safe init in [-0.5, 0]
        # After exp + negate: A in [-1.0, -0.6], keeping exp(dt*A) stable
        A_log_init = -torch.rand(d_inner, d_state) * 0.5
        self.A_log = nn.Parameter(A_log_init)

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sequential scan.

        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, d_model)
        """
        residual = x
        x = self.norm(x)
        B, T, _ = x.shape

        # Project input
        z = self.in_proj(x)  # (B, T, d_inner)
        gate = F.silu(self.gate_proj(x))  # (B, T, d_inner)

        # Compute input-dependent SSM parameters
        dt = F.softplus(self.dt_proj(z)).clamp(min=1e-3, max=1.0)  # (B, T, d_inner)
        B_param = self.B_proj(z)  # (B, T, d_state)
        C_param = self.C_proj(z)  # (B, T, d_state)

        # Sequential scan in fp32 for numerical stability
        # (Triton kernel available via ssm_scan_fn for inference)
        A = -self.A_log.exp().clamp(min=1e-6, max=1.0)  # A in [-1, -1e-6]
        d_inner = z.shape[-1]
        d_state = B_param.shape[-1]

        with torch.amp.autocast("cuda", enabled=False):
            z_f = z.float()
            dt_f = dt.float()
            B_f = B_param.float()
            C_f = C_param.float()
            A_f = A.float()

            h = torch.zeros(B, d_inner, d_state, device=x.device, dtype=torch.float32)
            outputs = []
            for t in range(T):
                dA = torch.exp(torch.clamp(dt_f[:, t, :, None] * A_f[None, :, :], min=-20.0, max=0.0))
                dB = dt_f[:, t, :, None] * B_f[:, t, None, :]
                h = dA * h + dB * z_f[:, t, :, None]
                y_t = (h * C_f[:, t, None, :]).sum(dim=-1)
                outputs.append(y_t)
            y = torch.stack(outputs, dim=1).to(x.dtype)

        # Gate and project
        y = y * gate
        y = self.out_proj(y)

        return residual + y


class MambaBlock(nn.Module):
    """Wrapper around mamba-ssm's Mamba block with pre-norm."""

    def __init__(self, d_model: int, d_state: int, expand: int = 2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))


class Level0SSM(nn.Module):
    """Level 0 dynamics: per-object SSM + global SSM conditioning.

    Each object gets its own SSM track (shared parameters).
    A separate global SSM processes the mean-pooled object states
    and conditions the per-object tracks.
    """

    def __init__(
        self, config: Level0Config, d_slot: int,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.d_slot = d_slot
        self.use_gradient_checkpointing = use_gradient_checkpointing

        BlockClass = MambaBlock if MAMBA_AVAILABLE else SimplifiedSSMBlock

        # Per-object SSM blocks
        self.object_blocks = nn.ModuleList([
            BlockClass(d_slot, config.d_state, config.expand_ratio)
            for _ in range(config.n_blocks)
        ])

        # Global SSM blocks
        self.global_blocks = nn.ModuleList([
            BlockClass(d_slot, config.d_state, config.expand_ratio)
            for _ in range(config.n_blocks)
        ])

        # Global -> per-object conditioning
        self.condition_proj = nn.Linear(d_slot, d_slot, bias=False)

    def forward(
        self,
        obj_states: torch.Tensor,
        obj_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run SSM dynamics on per-object temporal sequences.

        Args:
            obj_states: (B, N_max, T, d_slot) per-object state sequences.
            obj_mask: (B, N_max) which objects are alive.

        Returns:
            predicted: (B, N_max, T, d_slot) predicted states.
        """
        B, N, T, D = obj_states.shape

        # Global track: mean-pool across alive objects
        if obj_mask is not None:
            mask_expanded = obj_mask[:, :, None, None]  # (B, N, 1, 1)
            n_alive = obj_mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
            global_in = (obj_states * mask_expanded).sum(dim=1) / n_alive[:, :, None]
        else:
            global_in = obj_states.mean(dim=1)  # (B, T, D)

        # Run global SSM
        global_out = global_in
        for block in self.global_blocks:
            if self.use_gradient_checkpointing and self.training:
                global_out = torch.utils.checkpoint.checkpoint(
                    block, global_out, use_reentrant=False,
                )
            else:
                global_out = block(global_out)

        # Condition per-object: broadcast global context
        global_cond = self.condition_proj(global_out)  # (B, T, D)

        # Per-object SSM
        x = rearrange(obj_states, "b n t d -> (b n) t d")
        global_cond_expanded = global_cond.unsqueeze(1).expand(B, N, T, D)
        global_cond_expanded = rearrange(global_cond_expanded, "b n t d -> (b n) t d")
        x = x + global_cond_expanded

        for block in self.object_blocks:
            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, use_reentrant=False,
                )
            else:
                x = block(x)

        x = rearrange(x, "(b n) t d -> b n t d", b=B, n=N)

        # Clamp to prevent NaN/overflow — bf16 max is ~65504, stay well below
        x = torch.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)
        x = torch.clamp(x, min=-100.0, max=100.0)

        # Mask dead objects
        if obj_mask is not None:
            x = x * obj_mask[:, :, None, None]

        return x
