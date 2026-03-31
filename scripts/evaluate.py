"""Run HCLSM benchmark suite.

Usage:
    python scripts/evaluate.py --preset tiny
    python scripts/evaluate.py --config configs/hclsm_small.yaml --checkpoint checkpoints/step_10000.pt
"""

from __future__ import annotations

import argparse
import logging

import torch

from hclsm.config import HCLSMConfig
from hclsm.training.benchmarks import BenchmarkRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HCLSM world model")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--preset", type=str, default="tiny", choices=["tiny", "small", "base", "large"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.config:
        config = HCLSMConfig.from_yaml(args.config)
    else:
        config = getattr(HCLSMConfig, args.preset)()

    logger.info(f"Evaluating {args.preset} config on {args.device}")

    # Create model (lazy import to handle missing einops)
    try:
        from hclsm.model import HCLSMWorldModel
        model = HCLSMWorldModel(config)
        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"])
            logger.info(f"Loaded checkpoint from {args.checkpoint}")
        model = model.to(args.device)
    except ImportError as e:
        logger.error(f"Cannot create model: {e}")
        return

    # Run benchmarks
    runner = BenchmarkRunner()
    results = runner.run_all(model, device=args.device)

    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(runner.results_table(results))
    print("=" * 60)


if __name__ == "__main__":
    main()
