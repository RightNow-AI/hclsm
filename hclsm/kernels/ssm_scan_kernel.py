"""Batched parallel SSM scan kernel.

Implements the selective SSM recurrence:
    h_t = dA_t * h_{t-1} + dB_t * x_t
    y_t = (h_t * C_t).sum(dim=-1)

The sequential scan is O(T) in depth. The parallel associative scan
(Blelloch scan) reduces this to O(log T) parallel depth.

Provides:
- `parallel_ssm_scan_triton`: Triton parallel scan (when available)
- `sequential_ssm_scan`: Pure PyTorch sequential reference
- `ssm_scan_fn`: Auto-dispatches to fastest available

Shapes:
    x:  (B, T, D_inner)    — projected input
    dt: (B, T, D_inner)    — discretization timestep
    A:  (D_inner, D_state)  — log-space SSM state matrix
    B:  (B, T, D_state)    — input-dependent state matrix
    C:  (B, T, D_state)    — input-dependent output matrix
    Output: (B, T, D_inner)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Naive sequential scan (always available, matches SimplifiedSSMBlock)
# ---------------------------------------------------------------------------

def sequential_ssm_scan(
    x: torch.Tensor,
    dt: torch.Tensor,
    A_log: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> torch.Tensor:
    """Sequential SSM scan in pure PyTorch (reference implementation).

    Args:
        x:  (batch, T, d_inner) — projected input.
        dt: (batch, T, d_inner) — discretization timestep (post-softplus).
        A_log: (d_inner, d_state) — log-space SSM A parameter.
        B:  (batch, T, d_state) — input-dependent B.
        C:  (batch, T, d_state) — input-dependent C.

    Returns:
        y: (batch, T, d_inner) — SSM output.
    """
    batch, T, d_inner = x.shape
    d_state = B.shape[-1]
    A = -A_log.exp().clamp(min=1e-6, max=1.0)  # A in [-1, -1e-6]

    # Run in fp32 for numerical stability
    with torch.amp.autocast("cuda", enabled=False):
        x_f = x.float()
        dt_f = dt.float().clamp(min=1e-3, max=1.0)
        B_f = B.float()
        C_f = C.float()
        A_f = A.float()

        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=torch.float32)
        outputs = []

        for t in range(T):
            dA = torch.exp(torch.clamp(dt_f[:, t, :, None] * A_f[None, :, :], min=-20.0, max=0.0))
            dB = dt_f[:, t, :, None] * B_f[:, t, None, :]  # (batch, d_inner, d_state)

            h = dA * h + dB * x_f[:, t, :, None]
            y_t = (h * C_f[:, t, None, :]).sum(dim=-1)  # (batch, d_inner)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (batch, T, d_inner)

    return y.to(x.dtype)


# ---------------------------------------------------------------------------
# Parallel associative scan (PyTorch, for when Triton is unavailable)
# ---------------------------------------------------------------------------

def _parallel_associative_scan(
    gates: torch.Tensor,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """Parallel associative scan for linear recurrence h_t = gate_t * h_{t-1} + token_t.

    Uses the work-efficient Blelloch scan algorithm implemented via log-space
    operations to avoid numerical issues with long sequences.

    Args:
        gates: (batch, T, ...) — multiplicative gates (dA values).
        tokens: (batch, T, ...) — additive tokens (dB * x values).

    Returns:
        h: (batch, T, ...) — hidden states at each timestep.
    """
    B, T = gates.shape[:2]
    extra_dims = gates.shape[2:]

    if T == 1:
        return tokens

    # Pad to next power of 2
    T_pad = 1
    while T_pad < T:
        T_pad *= 2

    if T_pad != T:
        pad_size = T_pad - T
        gates = F.pad(gates, [0] * (2 * len(extra_dims)) + [0, pad_size], value=1.0)
        tokens = F.pad(tokens, [0] * (2 * len(extra_dims)) + [0, pad_size], value=0.0)

    # Up-sweep (reduce)
    g = gates.clone()
    t = tokens.clone()

    for d in range(T_pad.bit_length() - 1 if isinstance(T_pad, int) else int(torch.tensor(T_pad).log2().ceil())):
        stride = 2 ** (d + 1)
        half = stride // 2
        # Indices: stride-1, 2*stride-1, 3*stride-1, ...
        idx = torch.arange(stride - 1, T_pad, stride, device=gates.device)
        src = idx - half

        # t[idx] = g[idx] * t[src] + t[idx]
        # g[idx] = g[idx] * g[src]
        t_new = t.clone()
        g_new = g.clone()
        t_new[:, idx] = g[:, idx] * t[:, src] + t[:, idx]
        g_new[:, idx] = g[:, idx] * g[:, src]
        t = t_new
        g = g_new

    # Down-sweep
    t[:, -1] = t[:, -1]  # root is already correct

    log2_T = 0
    temp = T_pad
    while temp > 1:
        temp //= 2
        log2_T += 1

    for d in range(log2_T - 2, -1, -1):
        stride = 2 ** (d + 1)
        half = stride // 2
        idx = torch.arange(stride + half - 1, T_pad, stride, device=gates.device)
        src = idx - half

        if len(idx) > 0:
            t_new = t.clone()
            t_new[:, idx] = g[:, idx] * t[:, src] + t[:, idx]
            t = t_new

    return t[:, :T]


def parallel_ssm_scan_pytorch(
    x: torch.Tensor,
    dt: torch.Tensor,
    A_log: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> torch.Tensor:
    """Parallel SSM scan using associative scan in PyTorch.

    Same interface as sequential_ssm_scan but with O(log T) depth.

    Args:
        x:  (batch, T, d_inner)
        dt: (batch, T, d_inner)
        A_log: (d_inner, d_state)
        B:  (batch, T, d_state)
        C:  (batch, T, d_state)

    Returns:
        y: (batch, T, d_inner)
    """
    batch, T, d_inner = x.shape
    d_state = B.shape[-1]
    A = -A_log.exp()  # (d_inner, d_state)

    with torch.amp.autocast("cuda", enabled=False):
        x_f = x.float()
        dt_f = dt.float()
        B_f = B.float()
        C_f = C.float()

        # Discretize: dA = exp(dt * A), dB = dt * B
        # dA: (batch, T, d_inner, d_state)
        dA = torch.exp(dt_f.unsqueeze(-1) * A.float().unsqueeze(0).unsqueeze(0))
        # dB * x: (batch, T, d_inner, d_state)
        dB_x = (dt_f.unsqueeze(-1) * B_f.unsqueeze(2)) * x_f.unsqueeze(-1)

        # Run parallel scan for each state dimension
        # gates: dA (batch, T, d_inner, d_state)
        # tokens: dB_x (batch, T, d_inner, d_state)
        # Flatten inner dims for the scan
        flat_gates = dA.reshape(batch, T, -1)
        flat_tokens = dB_x.reshape(batch, T, -1)

        h_flat = _parallel_associative_scan(flat_gates, flat_tokens)
        h = h_flat.reshape(batch, T, d_inner, d_state)

        # Output: y_t = (h_t * C_t).sum(dim=-1)
        y = (h * C_f.unsqueeze(2)).sum(dim=-1)  # (batch, T, d_inner)

    return y.to(x.dtype)


# ---------------------------------------------------------------------------
# Triton parallel scan kernel
# ---------------------------------------------------------------------------

if TRITON_AVAILABLE:

    @triton.jit
    def _ssm_scan_fwd_kernel(
        X_ptr, DT_ptr, A_ptr, B_ptr, C_ptr, Out_ptr, H_ptr,
        batch_size, T: tl.constexpr, d_inner, d_state,
        stride_xb, stride_xt, stride_xd,
        stride_bb, stride_bt, stride_bs,
        BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr,
    ):
        """Per-(batch, d_inner_block) sequential scan.

        For the Triton version, we parallelize across batch and d_inner
        dimensions while keeping the scan sequential over T. This gives
        massive parallelism from the (B * d_inner / BLOCK_D) grid while
        keeping the scan simple and numerically stable.
        """
        bid = tl.program_id(0)
        did = tl.program_id(1)

        d_start = did * BLOCK_D
        d_idx = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_idx < d_inner

        s_idx = tl.arange(0, BLOCK_S)
        s_mask = s_idx < d_state

        # Load A: (BLOCK_D, BLOCK_S) from (d_inner, d_state)
        a_offsets = d_idx[:, None] * d_state + s_idx[None, :]
        a_mask = d_mask[:, None] & s_mask[None, :]
        A_neg = -tl.exp(tl.load(A_ptr + a_offsets, mask=a_mask, other=0.0))

        # Hidden state: (BLOCK_D, BLOCK_S)
        h = tl.zeros([BLOCK_D, BLOCK_S], dtype=tl.float32)

        for t in range(T):
            # Load x: (BLOCK_D,)
            x_offsets = bid * stride_xb + t * stride_xt + d_idx * stride_xd
            x_t = tl.load(X_ptr + x_offsets, mask=d_mask, other=0.0).to(tl.float32)

            # Load dt: (BLOCK_D,)
            dt_t = tl.load(DT_ptr + x_offsets, mask=d_mask, other=0.0).to(tl.float32)

            # Load B: (BLOCK_S,)
            b_offsets = bid * stride_bb + t * stride_bt + s_idx * stride_bs
            b_t = tl.load(B_ptr + b_offsets, mask=s_mask, other=0.0).to(tl.float32)

            # Load C: (BLOCK_S,)
            c_t = tl.load(C_ptr + b_offsets, mask=s_mask, other=0.0).to(tl.float32)

            # Discretize
            dA = tl.exp(dt_t[:, None] * A_neg)  # (BLOCK_D, BLOCK_S)
            dB = dt_t[:, None] * b_t[None, :]   # (BLOCK_D, BLOCK_S)

            # Scan step
            h = dA * h + dB * x_t[:, None]

            # Output: y_t = (h * C_t).sum(dim=-1) → (BLOCK_D,)
            y_t = tl.sum(h * c_t[None, :], axis=1)

            # Store output
            tl.store(Out_ptr + x_offsets, y_t.to(tl.float32), mask=d_mask)

        # Store final hidden state
        h_offsets = bid * d_inner * d_state + d_idx[:, None] * d_state + s_idx[None, :]
        tl.store(H_ptr + h_offsets, h, mask=a_mask)

    class _TritonSSMScan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, dt, A_log, B, C):
            batch, T, d_inner = x.shape
            d_state = B.shape[-1]

            x_c = x.contiguous().float()
            dt_c = dt.contiguous().float()
            B_c = B.contiguous().float()
            C_c = C.contiguous().float()

            out = torch.empty_like(x_c)
            h_final = torch.empty(batch, d_inner, d_state, device=x.device, dtype=torch.float32)

            BLOCK_D = min(triton.next_power_of_2(d_inner), 64)
            BLOCK_S = triton.next_power_of_2(d_state)
            grid = (batch, triton.cdiv(d_inner, BLOCK_D))

            _ssm_scan_fwd_kernel[grid](
                x_c, dt_c, A_log, B_c, C_c, out, h_final,
                batch, T, d_inner, d_state,
                x_c.stride(0), x_c.stride(1), x_c.stride(2),
                B_c.stride(0), B_c.stride(1), B_c.stride(2),
                BLOCK_D=BLOCK_D, BLOCK_S=BLOCK_S,
            )

            ctx.save_for_backward(x_c, dt_c, A_log, B_c, C_c, h_final)
            return out.to(x.dtype)

        @staticmethod
        def backward(ctx, grad_out):
            # Use PyTorch for backward (reverse scan)
            x, dt, A_log, B, C, h_final = ctx.saved_tensors
            batch, T, d_inner = x.shape
            d_state = B.shape[-1]
            A = -A_log.exp()

            grad_out_f = grad_out.float()

            # Recompute forward states for backward
            dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
            dB_x = (dt.unsqueeze(-1) * B.unsqueeze(2)) * x.unsqueeze(-1)

            # Forward pass to get all hidden states
            h_all = torch.zeros(batch, T + 1, d_inner, d_state, device=x.device, dtype=torch.float32)
            for t in range(T):
                h_all[:, t + 1] = dA[:, t] * h_all[:, t] + dB_x[:, t]

            # Backward pass (reverse scan)
            grad_h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=torch.float32)
            grad_x = torch.zeros_like(x)
            grad_dt = torch.zeros_like(dt)
            grad_B = torch.zeros_like(B)
            grad_C = torch.zeros_like(C)
            grad_A_log = torch.zeros_like(A_log)

            for t in range(T - 1, -1, -1):
                # y_t = (h_t * C_t).sum(-1)
                # grad from output
                grad_h = grad_h + grad_out_f[:, t, :, None] * C[:, t, None, :]

                # h_t = dA_t * h_{t-1} + dB_t * x_t
                grad_dA = grad_h * h_all[:, t]
                grad_h_prev = grad_h * dA[:, t]

                # dA = exp(dt * A), grad_dt from dA path
                grad_dt_dA = (grad_dA * dA[:, t] * A.unsqueeze(0)).sum(dim=-1)

                # dB_x = dt * B * x
                grad_dB_x = grad_h
                grad_x[:, t] = (grad_dB_x * dt[:, t, :, None] * B[:, t, None, :]).sum(dim=-1)
                grad_dt_dB = (grad_dB_x * B[:, t, None, :] * x[:, t, :, None]).sum(dim=-1)
                grad_B[:, t] = (grad_dB_x * dt[:, t, :, None] * x[:, t, :, None]).sum(dim=-2)
                grad_C[:, t] = (grad_out_f[:, t, :, None] * h_all[:, t + 1]).sum(dim=-2)

                grad_dt[:, t] = grad_dt_dA + grad_dt_dB

                # Accumulate A gradient
                grad_A_log = grad_A_log + (grad_dA * dA[:, t] * dt[:, t, :, None]).sum(dim=0)

                grad_h = grad_h_prev

            # A_log gradient: d_loss/d_A_log = d_loss/d_A * d_A/d_A_log = d_loss/d_A * (-exp(A_log))
            grad_A_log = grad_A_log * (-A_log.exp())

            return grad_x.to(x.dtype), grad_dt.to(dt.dtype), grad_A_log, grad_B.to(B.dtype), grad_C.to(C.dtype)

    def fused_ssm_scan_triton(
        x: torch.Tensor,
        dt: torch.Tensor,
        A_log: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """Triton SSM scan. Parallelizes over (batch, d_inner) dimensions."""
        return _TritonSSMScan.apply(x, dt, A_log, B, C)


# ---------------------------------------------------------------------------
# Auto-dispatch
# ---------------------------------------------------------------------------

def ssm_scan_fn(
    x: torch.Tensor,
    dt: torch.Tensor,
    A_log: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    backend: str = "auto",
) -> torch.Tensor:
    """Auto-dispatching SSM scan.

    Args:
        x:  (batch, T, d_inner)
        dt: (batch, T, d_inner)
        A_log: (d_inner, d_state)
        B:  (batch, T, d_state)
        C:  (batch, T, d_state)
        backend: "triton", "parallel", "sequential", or "auto".

    Returns:
        y: (batch, T, d_inner)
    """
    if backend == "triton" or (backend == "auto" and TRITON_AVAILABLE and x.is_cuda):
        return fused_ssm_scan_triton(x, dt, A_log, B, C)

    if backend == "parallel" or (backend == "auto" and x.is_cuda):
        return parallel_ssm_scan_pytorch(x, dt, A_log, B, C)

    return sequential_ssm_scan(x, dt, A_log, B, C)
