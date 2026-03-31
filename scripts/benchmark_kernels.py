"""Benchmark all HCLSM custom kernels.

Run on a GPU machine:
    python scripts/benchmark_kernels.py
    python scripts/benchmark_kernels.py --preset base --warmup 10 --trials 50
"""

from __future__ import annotations

import argparse
import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn

from hclsm.config import HCLSMConfig
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
from hclsm.kernels.gnn_message_pass import (
    naive_gnn_message_pass,
    optimized_gnn_message_pass,
    chunked_gnn_message_pass,
)


@dataclass
class BenchResult:
    name: str
    mean_ms: float
    std_ms: float
    speedup: float = 1.0


def benchmark_fn(fn, *args, warmup: int = 5, trials: int = 20, device: str = "cuda"):
    """Benchmark a function, returning mean and std in milliseconds."""
    # Warmup
    for _ in range(warmup):
        fn(*args)

    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(trials):
        if device == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn(*args)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            fn(*args)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    t = torch.tensor(times)
    return t.mean().item(), t.std().item()


def benchmark_slot_attention(config: HCLSMConfig, device: str, warmup: int, trials: int):
    """Benchmark slot attention variants."""
    print("\n" + "=" * 60)
    print("SLOT ATTENTION BENCHMARK")
    print("=" * 60)

    N = config.objects.n_max_slots
    D = config.objects.d_slot
    M = (config.perception.input_resolution // config.perception.patch_size) ** 2
    B = min(config.training.batch_size, 16)
    n_iter = config.objects.n_iterations

    print(f"  B={B}, N={N}, M={M}, D={D}, iterations={n_iter}")

    Q = torch.randn(B, N, D, device=device)
    K = torch.randn(B, M, D, device=device)
    V = torch.randn(B, M, D, device=device)

    results = []

    # Naive
    def run_naive():
        for _ in range(n_iter):
            naive_slot_attention(Q, K, V)

    mean, std = benchmark_fn(run_naive, warmup=warmup, trials=trials, device=device)
    naive_time = mean
    results.append(BenchResult("naive", mean, std))

    # Auto (Triton if available, else compiled)
    def run_auto():
        for _ in range(n_iter):
            slot_attention_fn(Q, K, V)

    mean, std = benchmark_fn(run_auto, warmup=warmup, trials=trials, device=device)
    results.append(BenchResult("auto", mean, std, naive_time / mean))

    for r in results:
        print(f"  {r.name:20s}: {r.mean_ms:8.2f}ms ± {r.std_ms:.2f}ms  (speedup: {r.speedup:.2f}x)")

    return results


def benchmark_ssm_scan(config: HCLSMConfig, device: str, warmup: int, trials: int):
    """Benchmark SSM scan variants."""
    print("\n" + "=" * 60)
    print("SSM SCAN BENCHMARK")
    print("=" * 60)

    B = min(config.training.batch_size, 8)
    N = config.objects.n_max_slots
    T = config.perception.temporal_resolution
    d_inner = config.objects.d_slot * config.dynamics.level0.expand_ratio
    d_state = config.dynamics.level0.d_state

    # Simulate batched objects: (B*N, T, d_inner)
    BN = B * N
    print(f"  B*N={BN}, T={T}, d_inner={d_inner}, d_state={d_state}")

    x = torch.randn(BN, T, d_inner, device=device)
    dt = torch.rand(BN, T, d_inner, device=device).abs() + 0.01
    A_log = torch.randn(d_inner, d_state, device=device)
    Bp = torch.randn(BN, T, d_state, device=device)
    C = torch.randn(BN, T, d_state, device=device)

    results = []

    # Sequential
    mean, std = benchmark_fn(sequential_ssm_scan, x, dt, A_log, Bp, C,
                             warmup=warmup, trials=trials, device=device)
    seq_time = mean
    results.append(BenchResult("sequential", mean, std))

    # Parallel (PyTorch)
    mean, std = benchmark_fn(parallel_ssm_scan_pytorch, x, dt, A_log, Bp, C,
                             warmup=warmup, trials=trials, device=device)
    results.append(BenchResult("parallel_pytorch", mean, std, seq_time / mean))

    # Auto
    mean, std = benchmark_fn(ssm_scan_fn, x, dt, A_log, Bp, C,
                             warmup=warmup, trials=trials, device=device)
    results.append(BenchResult("auto", mean, std, seq_time / mean))

    for r in results:
        print(f"  {r.name:20s}: {r.mean_ms:8.2f}ms ± {r.std_ms:.2f}ms  (speedup: {r.speedup:.2f}x)")

    return results


def benchmark_gnn(config: HCLSMConfig, device: str, warmup: int, trials: int):
    """Benchmark GNN message passing variants."""
    print("\n" + "=" * 60)
    print("GNN MESSAGE PASSING BENCHMARK")
    print("=" * 60)

    B = min(config.training.batch_size, 16)
    N = config.objects.n_max_slots
    D = config.objects.d_slot
    d_edge = config.objects.d_edge
    n_rounds = config.objects.gnn_rounds

    print(f"  B={B}, N={N}, D={D}, d_edge={d_edge}, rounds={n_rounds}")

    nodes = torch.randn(B, N, D, device=device)
    alive_mask = torch.ones(B, N, device=device)

    edge_mlp = nn.Sequential(
        nn.Linear(D * 4, d_edge * 2), nn.ReLU(),
        nn.Linear(d_edge * 2, d_edge), nn.ReLU(),
    ).to(device)
    edge_weight_linear = nn.Linear(d_edge, 1).to(device)
    msg_mlp = nn.Sequential(nn.Linear(d_edge, D), nn.ReLU()).to(device)
    update_mlp = nn.Sequential(
        nn.Linear(D * 2, D), nn.ReLU(), nn.Linear(D, D),
    ).to(device)

    results = []

    # Naive
    def run_naive():
        return naive_gnn_message_pass(
            nodes, alive_mask, edge_mlp, edge_weight_linear,
            msg_mlp, update_mlp, n_rounds,
        )

    mean, std = benchmark_fn(run_naive, warmup=warmup, trials=trials, device=device)
    naive_time = mean
    results.append(BenchResult("naive", mean, std))

    # Optimized
    def run_optimized():
        return optimized_gnn_message_pass(
            nodes, alive_mask, edge_mlp, edge_weight_linear,
            msg_mlp, update_mlp, n_rounds,
        )

    mean, std = benchmark_fn(run_optimized, warmup=warmup, trials=trials, device=device)
    results.append(BenchResult("optimized", mean, std, naive_time / mean))

    # Chunked
    def run_chunked():
        return chunked_gnn_message_pass(
            nodes, alive_mask, edge_mlp, edge_weight_linear,
            msg_mlp, update_mlp, n_rounds, chunk_size=16,
        )

    mean, std = benchmark_fn(run_chunked, warmup=warmup, trials=trials, device=device)
    results.append(BenchResult("chunked_16", mean, std, naive_time / mean))

    for r in results:
        print(f"  {r.name:20s}: {r.mean_ms:8.2f}ms ± {r.std_ms:.2f}ms  (speedup: {r.speedup:.2f}x)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark HCLSM kernels")
    parser.add_argument("--preset", default="tiny", choices=["tiny", "small", "base", "large"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()

    config = getattr(HCLSMConfig, args.preset)()
    device = args.device

    print(f"Benchmarking {args.preset} config on {device}")
    print(f"Triton available: {TRITON_AVAILABLE}")

    benchmark_slot_attention(config, device, args.warmup, args.trials)
    benchmark_ssm_scan(config, device, args.warmup, args.trials)
    benchmark_gnn(config, device, args.warmup, args.trials)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
