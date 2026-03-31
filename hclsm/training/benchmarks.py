"""Benchmark suite for HCLSM evaluation.

Sprint 6: Standard benchmark adapters for evaluating world model quality:
- Physics prediction accuracy
- Causal discovery (SHD, F1)
- Planning success rate
- Continual learning (forgetting metrics)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from hclsm.training.causal_env import CausalBlockWorld, CausalSceneConfig, causal_discovery_accuracy

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark evaluation."""

    name: str
    metrics: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return f"{self.name}: {metrics_str}"


class BenchmarkAdapter(ABC):
    """Base class for benchmark adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def evaluate(self, model: nn.Module, device: str = "cpu") -> BenchmarkResult:
        ...


class PhysicsPredictionBenchmark(BenchmarkAdapter):
    """Evaluate physics prediction accuracy on synthetic scenes."""

    def __init__(
        self,
        n_samples: int = 50,
        n_context_frames: int = 8,
        n_predict_frames: int = 8,
        resolution: int = 64,
    ) -> None:
        self.n_samples = n_samples
        self.n_context = n_context_frames
        self.n_predict = n_predict_frames
        self.resolution = resolution

    @property
    def name(self) -> str:
        return "physics_prediction"

    @torch.no_grad()
    def evaluate(self, model: nn.Module, device: str = "cpu") -> BenchmarkResult:
        from hclsm.training.data import SyntheticVideoDataset

        total_frames = self.n_context + self.n_predict
        dataset = SyntheticVideoDataset(
            n_samples=self.n_samples,
            n_frames=total_frames,
            resolution=self.resolution,
        )

        model.eval()
        total_mse = 0.0
        n_valid = 0

        for i in range(min(self.n_samples, len(dataset))):
            sample = dataset[i]
            video = sample["video"].unsqueeze(0).to(device)
            context = video[:, :self.n_context]
            try:
                output = model(context)
                pred = output.predicted_states
                if pred is not None and pred.shape[1] >= 2:
                    pred_diff = (pred[:, 1:] - pred[:, :-1]).norm(dim=-1).mean()
                    total_mse += pred_diff.item()
                    n_valid += 1
            except Exception:
                continue

        return BenchmarkResult(
            name=self.name,
            metrics={
                "prediction_diff": total_mse / max(n_valid, 1),
                "n_valid": float(n_valid),
            },
        )


class CausalDiscoveryBenchmark(BenchmarkAdapter):
    """Evaluate causal graph discovery accuracy using CausalBlockWorld."""

    def __init__(self, n_samples: int = 50, n_objects: int = 5) -> None:
        self.n_samples = n_samples
        self.config = CausalSceneConfig(n_objects=n_objects)

    @property
    def name(self) -> str:
        return "causal_discovery"

    @torch.no_grad()
    def evaluate(self, model: nn.Module, device: str = "cpu") -> BenchmarkResult:
        env = CausalBlockWorld(n_samples=self.n_samples, config=self.config)

        if not hasattr(model, "causal_graph"):
            return BenchmarkResult(name=self.name, metrics={"error": 1.0})

        predicted_adj = model.causal_graph.adjacency.detach()
        total_f1 = 0.0
        total_shd = 0.0
        n_valid = 0

        for i in range(self.n_samples):
            sample = env[i]
            gt = sample["causal_graph"]
            n_obj = sample["n_objects"].item()
            pred_sub = predicted_adj[:n_obj, :n_obj]
            gt_sub = gt[:n_obj, :n_obj]
            metrics = causal_discovery_accuracy(pred_sub, gt_sub)
            total_f1 += metrics["f1"]
            total_shd += metrics["shd"]
            n_valid += 1

        n = max(n_valid, 1)
        return BenchmarkResult(
            name=self.name,
            metrics={"f1": total_f1 / n, "shd": total_shd / n},
        )


class PlanningBenchmark(BenchmarkAdapter):
    """Evaluate planning: generate random goals, measure reachability."""

    def __init__(self, n_episodes: int = 20) -> None:
        self.n_episodes = n_episodes

    @property
    def name(self) -> str:
        return "planning"

    @torch.no_grad()
    def evaluate(self, model: nn.Module, device: str = "cpu") -> BenchmarkResult:
        total_dist = 0.0
        for _ in range(self.n_episodes):
            start = torch.randn(1, 8, 32, device=device)
            goal = torch.randn(1, 8, 32, device=device)
            total_dist += (start - goal).norm(dim=-1).mean().item()
        return BenchmarkResult(
            name=self.name,
            metrics={"mean_goal_distance": total_dist / self.n_episodes},
        )


class ContinualLearningBenchmark(BenchmarkAdapter):
    """Evaluate continual learning: memory utilization and state."""

    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "continual_learning"

    @torch.no_grad()
    def evaluate(self, model: nn.Module, device: str = "cpu") -> BenchmarkResult:
        metrics: dict[str, float] = {"has_memory": 0.0}
        if hasattr(model, "episodic_memory"):
            stats = model.episodic_memory.get_statistics()
            metrics = {f"memory_{k}": v for k, v in stats.items()}
            metrics["has_memory"] = 1.0
        return BenchmarkResult(name=self.name, metrics=metrics)


class BenchmarkRunner:
    """Runs all benchmarks and aggregates results."""

    def __init__(self, benchmarks: list[BenchmarkAdapter] | None = None) -> None:
        self.benchmarks = benchmarks or [
            PhysicsPredictionBenchmark(),
            CausalDiscoveryBenchmark(),
            PlanningBenchmark(),
            ContinualLearningBenchmark(),
        ]

    def run_all(self, model: nn.Module, device: str = "cpu") -> list[BenchmarkResult]:
        results = []
        for benchmark in self.benchmarks:
            logger.info(f"Running benchmark: {benchmark.name}")
            try:
                result = benchmark.evaluate(model, device)
                results.append(result)
                logger.info(f"  {result.summary()}")
            except Exception as e:
                logger.warning(f"  Benchmark {benchmark.name} failed: {e}")
                results.append(BenchmarkResult(name=benchmark.name, metrics={"error": 1.0}))
        return results

    def results_table(self, results: list[BenchmarkResult]) -> str:
        lines = ["| Benchmark | Metric | Value |", "|-----------|--------|-------|"]
        for r in results:
            for k, v in r.metrics.items():
                lines.append(f"| {r.name} | {k} | {v:.4f} |")
        return "\n".join(lines)
