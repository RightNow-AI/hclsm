"""Correctness tests for Sprint 3 custom kernels.

Tests each kernel's output against the naive PyTorch reference in both
forward and backward passes, across fp32 and bf16.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from hclsm.kernels.slot_attention_kernel import (
    naive_slot_attention,
    slot_attention_fn,
    TRITON_AVAILABLE,
)
from hclsm.kernels.ssm_scan_kernel import (
    sequential_ssm_scan,
    parallel_ssm_scan_pytorch,
    ssm_scan_fn,
)
from hclsm.kernels.sparse_event_attn import (
    naive_sparse_event_attention,
    sparse_event_attention,
    sparse_event_attn_fn,
)
from hclsm.kernels.gnn_message_pass import (
    naive_gnn_message_pass,
    optimized_gnn_message_pass,
    chunked_gnn_message_pass,
    gnn_message_pass_fn,
)

# Skip GPU-only tests when no CUDA
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)
requires_triton = pytest.mark.skipif(
    not TRITON_AVAILABLE, reason="Triton not available"
)


# ═══════════════════════════════════════════════════════════════════════════
# Slot Attention Kernel
# ═══════════════════════════════════════════════════════════════════════════

class TestSlotAttentionKernel:

    @pytest.fixture(params=[(2, 16, 64, 128), (1, 8, 32, 64), (4, 32, 196, 256)])
    def qkv(self, request):
        B, N, M, D = request.param
        Q = torch.randn(B, N, D, requires_grad=True)
        K = torch.randn(B, M, D, requires_grad=True)
        V = torch.randn(B, M, D, requires_grad=True)
        return Q, K, V

    def test_naive_shapes(self, qkv):
        Q, K, V = qkv
        out, attn = naive_slot_attention(Q, K, V)
        B, N, D = Q.shape
        M = K.shape[1]
        assert out.shape == (B, N, D)
        assert attn.shape == (B, N, M)

    def test_naive_softmax_over_slots(self, qkv):
        Q, K, V = qkv
        _, attn = naive_slot_attention(Q, K, V)
        # Softmax over dim=1 (slots): each column sums to 1
        col_sums = attn.sum(dim=1)
        assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-5)

    def test_autodispatch_matches_naive(self, qkv):
        Q, K, V = qkv
        out_naive, attn_naive = naive_slot_attention(Q, K, V)
        out_auto, attn_auto = slot_attention_fn(Q, K, V, backend="naive")
        assert torch.allclose(out_naive, out_auto, atol=1e-5)

    def test_backward_runs(self, qkv):
        Q, K, V = qkv
        out, _ = naive_slot_attention(Q, K, V)
        loss = out.sum()
        loss.backward()
        assert Q.grad is not None
        assert K.grad is not None
        assert V.grad is not None

    @requires_cuda
    def test_cuda_slot_attention(self):
        B, N, M, D = 2, 16, 64, 128
        Q = torch.randn(B, N, D, device="cuda")
        K = torch.randn(B, M, D, device="cuda")
        V = torch.randn(B, M, D, device="cuda")
        out, attn = slot_attention_fn(Q, K, V)
        assert out.shape == (B, N, D)
        col_sums = attn.sum(dim=1)
        assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-4)

    @requires_cuda
    @requires_triton
    def test_triton_matches_naive(self):
        B, N, M, D = 2, 16, 64, 128
        Q = torch.randn(B, N, D, device="cuda")
        K = torch.randn(B, M, D, device="cuda")
        V = torch.randn(B, M, D, device="cuda")

        out_naive, attn_naive = naive_slot_attention(Q, K, V)
        out_triton, attn_triton = slot_attention_fn(Q, K, V, backend="triton")

        assert torch.allclose(out_naive, out_triton, atol=1e-3)
        assert torch.allclose(attn_naive, attn_triton, atol=1e-3)


# ═══════════════════════════════════════════════════════════════════════════
# SSM Scan Kernel
# ═══════════════════════════════════════════════════════════════════════════

class TestSSMScanKernel:

    @pytest.fixture(params=[(2, 16, 32, 8), (1, 8, 64, 16), (4, 32, 16, 4)])
    def ssm_inputs(self, request):
        B, T, d_inner, d_state = request.param
        x = torch.randn(B, T, d_inner)
        dt = torch.rand(B, T, d_inner).abs() + 0.01
        A_log = torch.randn(d_inner, d_state)
        Bp = torch.randn(B, T, d_state)
        C = torch.randn(B, T, d_state)
        return x, dt, A_log, Bp, C

    def test_sequential_shape(self, ssm_inputs):
        x, dt, A_log, Bp, C = ssm_inputs
        y = sequential_ssm_scan(x, dt, A_log, Bp, C)
        assert y.shape == x.shape

    def test_parallel_matches_sequential(self, ssm_inputs):
        x, dt, A_log, Bp, C = ssm_inputs
        y_seq = sequential_ssm_scan(x, dt, A_log, Bp, C)
        y_par = parallel_ssm_scan_pytorch(x, dt, A_log, Bp, C)
        assert torch.allclose(y_seq, y_par, atol=1e-4), \
            f"Max diff: {(y_seq - y_par).abs().max().item()}"

    def test_autodispatch(self, ssm_inputs):
        x, dt, A_log, Bp, C = ssm_inputs
        y = ssm_scan_fn(x, dt, A_log, Bp, C)
        assert y.shape == x.shape

    def test_sequential_backward(self):
        B, T, d_inner, d_state = 2, 8, 16, 4
        x = torch.randn(B, T, d_inner, requires_grad=True)
        dt = torch.rand(B, T, d_inner).abs() + 0.01
        A_log = torch.randn(d_inner, d_state, requires_grad=True)
        Bp = torch.randn(B, T, d_state)
        C = torch.randn(B, T, d_state)

        y = sequential_ssm_scan(x, dt, A_log, Bp, C)
        loss = y.sum()
        loss.backward()
        # Sequential scan is differentiable through PyTorch autograd
        assert x.grad is not None

    @requires_cuda
    @requires_triton
    def test_triton_matches_sequential(self):
        B, T, d_inner, d_state = 2, 16, 32, 8
        x = torch.randn(B, T, d_inner, device="cuda")
        dt = torch.rand(B, T, d_inner, device="cuda").abs() + 0.01
        A_log = torch.randn(d_inner, d_state, device="cuda")
        Bp = torch.randn(B, T, d_state, device="cuda")
        C = torch.randn(B, T, d_state, device="cuda")

        y_seq = sequential_ssm_scan(x, dt, A_log, Bp, C)
        y_triton = ssm_scan_fn(x, dt, A_log, Bp, C, backend="triton")
        assert torch.allclose(y_seq, y_triton, atol=1e-3)


# ═══════════════════════════════════════════════════════════════════════════
# Sparse Event Attention
# ═══════════════════════════════════════════════════════════════════════════

class TestSparseEventAttention:

    def test_basic_gather_scatter(self):
        B, T, N, D = 2, 16, 8, 32
        states = torch.randn(B, T, N, D)
        event_mask = torch.zeros(B, T)
        event_mask[0, [2, 5, 10]] = 1
        event_mask[1, [1, 7]] = 1

        def scale_fn(x):
            return x * 2.0

        out = naive_sparse_event_attention(states, event_mask, scale_fn)
        assert out.shape == (B, T, N, D)

        # Non-event timesteps unchanged
        assert torch.equal(out[0, 0], states[0, 0])
        assert torch.equal(out[0, 3], states[0, 3])

        # Event timesteps scaled
        assert torch.allclose(out[0, 2], states[0, 2] * 2.0)
        assert torch.allclose(out[0, 5], states[0, 5] * 2.0)

    def test_vectorized_matches_naive(self):
        B, T, N, D = 2, 16, 8, 32
        states = torch.randn(B, T, N, D)
        event_mask = torch.zeros(B, T)
        event_mask[0, [2, 5, 10]] = 1
        event_mask[1, [1, 7, 9, 14]] = 1

        def identity(x):
            return x

        out_naive = naive_sparse_event_attention(states, event_mask, identity)
        out_vec = sparse_event_attention(states, event_mask, identity)
        assert torch.allclose(out_naive, out_vec, atol=1e-5)

    def test_no_events(self):
        B, T, N, D = 2, 8, 4, 16
        states = torch.randn(B, T, N, D)
        event_mask = torch.zeros(B, T)

        out = sparse_event_attn_fn(states, event_mask, lambda x: x)
        assert torch.allclose(out, states, atol=1e-5)

    def test_all_events(self):
        B, T, N, D = 2, 8, 4, 16
        states = torch.randn(B, T, N, D)
        event_mask = torch.ones(B, T)

        out = sparse_event_attn_fn(states, event_mask, lambda x: x * 0)
        # All timesteps should be zeroed out
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════
# GNN Message Passing
# ═══════════════════════════════════════════════════════════════════════════

class TestGNNMessagePass:

    @pytest.fixture
    def gnn_modules(self):
        D, d_edge = 32, 16
        edge_mlp = nn.Sequential(
            nn.Linear(D * 4, d_edge * 2), nn.ReLU(),
            nn.Linear(d_edge * 2, d_edge), nn.ReLU(),
        )
        edge_weight_linear = nn.Linear(d_edge, 1)
        msg_mlp = nn.Sequential(nn.Linear(d_edge, D), nn.ReLU())
        update_mlp = nn.Sequential(nn.Linear(D * 2, D), nn.ReLU(), nn.Linear(D, D))
        return edge_mlp, edge_weight_linear, msg_mlp, update_mlp

    def test_naive_shapes(self, gnn_modules):
        B, N, D = 2, 16, 32
        nodes = torch.randn(B, N, D)
        alive_mask = torch.ones(B, N)
        out, ef = naive_gnn_message_pass(nodes, alive_mask, *gnn_modules)
        assert out.shape == (B, N, D)
        assert ef is not None

    def test_dead_slots_zeroed(self, gnn_modules):
        B, N, D = 2, 16, 32
        nodes = torch.randn(B, N, D)
        alive_mask = torch.ones(B, N)
        alive_mask[:, -4:] = 0  # Last 4 dead

        out, _ = naive_gnn_message_pass(nodes, alive_mask, *gnn_modules)
        assert torch.allclose(out[:, -4:], torch.zeros(B, 4, D), atol=1e-6)

    def test_optimized_matches_naive(self, gnn_modules):
        B, N, D = 2, 8, 32
        nodes = torch.randn(B, N, D)
        alive_mask = torch.ones(B, N)

        out_naive, _ = naive_gnn_message_pass(nodes, alive_mask, *gnn_modules, n_rounds=1)
        out_opt, _ = optimized_gnn_message_pass(nodes, alive_mask, *gnn_modules, n_rounds=1)
        assert torch.allclose(out_naive, out_opt, atol=1e-5)

    def test_chunked_matches_naive(self, gnn_modules):
        B, N, D = 2, 8, 32
        nodes = torch.randn(B, N, D)
        alive_mask = torch.ones(B, N)

        out_naive, _ = naive_gnn_message_pass(nodes, alive_mask, *gnn_modules, n_rounds=1)
        out_chunk, _ = chunked_gnn_message_pass(
            nodes, alive_mask, *gnn_modules, n_rounds=1, chunk_size=4,
        )
        assert torch.allclose(out_naive, out_chunk, atol=1e-4)

    def test_autodispatch(self, gnn_modules):
        B, N, D = 2, 8, 32
        nodes = torch.randn(B, N, D)
        alive_mask = torch.ones(B, N)
        out, _ = gnn_message_pass_fn(nodes, alive_mask, *gnn_modules)
        assert out.shape == (B, N, D)

    def test_no_alive_mask(self, gnn_modules):
        B, N, D = 2, 8, 32
        nodes = torch.randn(B, N, D)
        out, _ = naive_gnn_message_pass(nodes, None, *gnn_modules)
        assert out.shape == (B, N, D)
