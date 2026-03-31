"""Fused slot attention kernel.

Fuses Q@K^T → scale → softmax(dim=1 over slots) → @V into a single kernel.
CRITICAL: The softmax is over the SLOT dimension (dim=1), NOT the key dimension.
This creates competition between slots for input tokens.

Provides:
- `fused_slot_attention`: Triton kernel (when available)
- `naive_slot_attention`: Pure PyTorch reference
- `slot_attention_fn`: Auto-dispatches to fastest available

Shapes:
    Q: (B, N, D)   — slot queries
    K: (B, M, D)   — input keys
    V: (B, M, D)   — input values
    scale: float    — 1/sqrt(D) typically
    Output: (B, N, D), attn_weights: (B, N, M)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Triton kernel (available on Linux with CUDA GPU + triton installed)
# ---------------------------------------------------------------------------

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _fused_slot_attn_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, Out_ptr, Attn_ptr,
        scale,
        N: tl.constexpr, M_total, D: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """Forward kernel: one program per (batch, m_block).

        Two-pass approach to handle large M within shared memory limits:
        Pass 1: Compute logits for this M block + online softmax stats
        Pass 2: (in second grid) Finalize softmax + compute output

        Simplified: since N is small (<=128), we process all N at once
        and tile over M blocks. Each block computes partial logits,
        and we store them to the attention buffer for later use.
        """
        bid = tl.program_id(0)
        mid = tl.program_id(1)

        m_start = mid * BLOCK_M
        n_idx = tl.arange(0, N)
        m_idx = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_idx < M_total

        q_base = Q_ptr + bid * N * D
        k_base = K_ptr + bid * M_total * D
        attn_base = Attn_ptr + bid * N * M_total

        # Compute logits[N, BLOCK_M] via tiled matmul over D
        logits = tl.zeros([N, BLOCK_M], dtype=tl.float32)

        for d_start in range(0, D, BLOCK_D):
            d_idx = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_idx < D

            q_offsets = n_idx[:, None] * D + d_idx[None, :]
            q_tile = tl.load(q_base + q_offsets, mask=d_mask[None, :], other=0.0)

            k_offsets = m_idx[:, None] * D + d_idx[None, :]
            k_tile = tl.load(k_base + k_offsets, mask=(m_mask[:, None] & d_mask[None, :]), other=0.0)

            logits += tl.dot(q_tile, tl.trans(k_tile))

        logits = logits * scale

        # Mask out-of-bounds M positions
        logits = tl.where(m_mask[None, :], logits, float('-inf'))

        # Store raw logits to attention buffer (will be normalized in second pass)
        attn_offsets = n_idx[:, None] * M_total + m_idx[None, :]
        tl.store(attn_base + attn_offsets, logits, mask=m_mask[None, :])

    @triton.jit
    def _slot_attn_softmax_output_kernel(
        Attn_ptr, V_ptr, Out_ptr,
        N: tl.constexpr, M_total, D: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """Normalize attention (column-wise softmax over N) and compute output.

        Two phases within one kernel:
        Phase A: Softmax-normalize all M blocks (reads raw logits, writes normalized attn)
        Phase B: Compute output = attn @ V, tiled over D and M
        """
        bid = tl.program_id(0)

        n_idx = tl.arange(0, N)
        attn_base = Attn_ptr + bid * N * M_total
        v_base = V_ptr + bid * M_total * D
        out_base = Out_ptr + bid * N * D

        # ── Phase A: Softmax normalization (one pass over M) ──
        for m_start in range(0, M_total, BLOCK_M):
            m_idx = m_start + tl.arange(0, BLOCK_M)
            m_mask = m_idx < M_total

            attn_offsets = n_idx[:, None] * M_total + m_idx[None, :]
            logits_block = tl.load(attn_base + attn_offsets, mask=m_mask[None, :], other=float('-inf'))

            # Column-wise softmax over N (axis=0)
            col_max = tl.max(logits_block, axis=0)
            logits_block = logits_block - col_max[None, :]
            exp_block = tl.exp(logits_block)
            col_sum = tl.sum(exp_block, axis=0) + 1e-8
            attn_block = exp_block / col_sum[None, :]

            tl.store(attn_base + attn_offsets, attn_block, mask=m_mask[None, :])

        # ── Phase B: Output = attn @ V, tiled over D and M ──
        for d_start in range(0, D, BLOCK_D):
            d_idx = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_idx < D
            out_acc = tl.zeros([N, BLOCK_D], dtype=tl.float32)

            for m_start in range(0, M_total, BLOCK_M):
                m_idx = m_start + tl.arange(0, BLOCK_M)
                m_mask = m_idx < M_total

                # Load already-normalized attention: (N, BLOCK_M)
                attn_offsets = n_idx[:, None] * M_total + m_idx[None, :]
                attn_block = tl.load(attn_base + attn_offsets, mask=m_mask[None, :], other=0.0)

                # Load V block: (BLOCK_M, BLOCK_D)
                v_offsets = m_idx[:, None] * D + d_idx[None, :]
                v_tile = tl.load(v_base + v_offsets, mask=(m_mask[:, None] & d_mask[None, :]), other=0.0)

                out_acc += tl.dot(attn_block, v_tile)

            out_offsets = n_idx[:, None] * D + d_idx[None, :]
            tl.store(out_base + out_offsets, out_acc, mask=d_mask[None, :])

    class _FusedSlotAttentionTriton(torch.autograd.Function):
        """Autograd wrapper for the Triton fused slot attention kernel."""

        @staticmethod
        def forward(
            ctx,
            Q: torch.Tensor,
            K: torch.Tensor,
            V: torch.Tensor,
            scale: float,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            B, N, D = Q.shape
            M = K.shape[1]

            N_pad = triton.next_power_of_2(N)

            if N != N_pad:
                Q = F.pad(Q, (0, 0, 0, N_pad - N))

            Q_c = Q.contiguous()
            K_c = K.contiguous()
            V_c = V.contiguous()

            Out = torch.empty(B, N_pad, D, device=Q.device, dtype=Q.dtype)
            Attn = torch.empty(B, N_pad, M, device=Q.device, dtype=torch.float32)

            # Choose block sizes to fit in shared memory
            # Budget: ~48KB (safe for T4's 64KB with headroom)
            # Per M block: N*BLOCK_M (logits) + N*BLOCK_D (Q) + BLOCK_M*BLOCK_D (K)
            BLOCK_D = min(triton.next_power_of_2(D), 32)
            BLOCK_M = min(triton.next_power_of_2(M), 64)

            # Ensure BLOCK_M * N * 4 bytes fits in shared memory
            while BLOCK_M * N_pad * 4 + BLOCK_M * BLOCK_D * 4 > 40000 and BLOCK_M > 16:
                BLOCK_M //= 2

            n_m_blocks = triton.cdiv(M, BLOCK_M)

            # Pass 1: Compute logits blocks
            _fused_slot_attn_fwd_kernel[(B, n_m_blocks)](
                Q_c, K_c, V_c, Out, Attn,
                scale,
                N=N_pad, M_total=M, D=D,
                BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
            )

            # Pass 2: Softmax + output
            _slot_attn_softmax_output_kernel[(B,)](
                Attn, V_c, Out,
                N=N_pad, M_total=M, D=D,
                BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
            )

            Out = Out[:, :N, :]
            Attn = Attn[:, :N, :]

            ctx.save_for_backward(Q[:, :N, :], K, V, Attn)
            ctx.scale = scale
            return Out, Attn

        @staticmethod
        def backward(ctx, grad_out, grad_attn):
            Q, K, V, attn = ctx.saved_tensors
            scale = ctx.scale

            # d_attn from output path: grad_out @ V^T → (B, N, M)
            d_attn = torch.bmm(grad_out, V.transpose(1, 2))

            # Add direct gradient on attention weights
            if grad_attn is not None:
                d_attn = d_attn + grad_attn

            # Softmax backward (dim=0 / column-wise):
            # d_logits[n,m] = attn[n,m] * (d_attn[n,m] - sum_n'(attn[n',m] * d_attn[n',m]))
            sum_da = (attn * d_attn).sum(dim=1, keepdim=True)  # (B, 1, M)
            d_logits = attn * (d_attn - sum_da) * scale

            # Matmul backward
            d_Q = torch.bmm(d_logits, K)       # (B, N, D)
            d_K = torch.bmm(d_logits.transpose(1, 2), Q)  # (B, M, D)
            d_V = torch.bmm(attn.transpose(1, 2), grad_out)  # (B, M, D)

            return d_Q, d_K, d_V, None

    def fused_slot_attention_triton(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Triton fused slot attention.

        Args:
            Q: (B, N, D) slot queries.
            K: (B, M, D) input keys.
            V: (B, M, D) input values.
            scale: Scaling factor (default: 1/sqrt(D)).

        Returns:
            output: (B, N, D) weighted sum of values.
            attn: (B, N, M) attention weights (softmax over N for each M).
        """
        if scale is None:
            scale = Q.shape[-1] ** -0.5
        return _FusedSlotAttentionTriton.apply(Q, K, V, scale)


# ---------------------------------------------------------------------------
# Naive PyTorch reference (always available)
# ---------------------------------------------------------------------------

def naive_slot_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch slot attention reference.

    CRITICAL: softmax is over dim=1 (SLOT dimension), NOT dim=2 (key dimension).

    Args:
        Q: (B, N, D) slot queries.
        K: (B, M, D) input keys.
        V: (B, M, D) input values.
        scale: Scaling factor (default: 1/sqrt(D)).

    Returns:
        output: (B, N, D) weighted sum of values.
        attn: (B, N, M) attention weights.
    """
    if scale is None:
        scale = Q.shape[-1] ** -0.5

    logits = torch.bmm(Q, K.transpose(1, 2)) * scale  # (B, N, M)
    attn = F.softmax(logits, dim=1)  # softmax over SLOTS
    output = torch.bmm(attn, V)  # (B, N, D)
    return output, attn


# ---------------------------------------------------------------------------
# torch.compile optimized version
# ---------------------------------------------------------------------------

@torch.compile(fullgraph=True, dynamic=False)
def _compiled_slot_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.bmm(Q, K.transpose(1, 2)) * scale
    attn = F.softmax(logits, dim=1)
    output = torch.bmm(attn, V)
    return output, attn


def compiled_slot_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """torch.compile fused slot attention (middle ground)."""
    if scale is None:
        scale = Q.shape[-1] ** -0.5
    return _compiled_slot_attention(Q, K, V, scale)


# ---------------------------------------------------------------------------
# Auto-dispatch
# ---------------------------------------------------------------------------

def slot_attention_fn(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float | None = None,
    backend: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Auto-dispatching slot attention.

    Args:
        Q: (B, N, D) slot queries.
        K: (B, M, D) input keys.
        V: (B, M, D) input values.
        scale: Scaling factor (default: 1/sqrt(D)).
        backend: "triton", "compiled", "naive", or "auto".

    Returns:
        output: (B, N, D), attn: (B, N, M).
    """
    if scale is None:
        scale = Q.shape[-1] ** -0.5

    if backend == "triton" or (backend == "auto" and TRITON_AVAILABLE and Q.is_cuda):
        return fused_slot_attention_triton(Q, K, V, scale)

    if backend == "compiled" or (backend == "auto" and Q.is_cuda):
        try:
            return compiled_slot_attention(Q, K, V, scale)
        except Exception:
            pass

    return naive_slot_attention(Q, K, V, scale)
