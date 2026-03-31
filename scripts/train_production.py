"""Production training run for HCLSM.

Trains with full metrics: loss curves, causal discovery accuracy, slot stats.

Usage:
    python scripts/train_production.py --preset small --steps 50000
    python scripts/train_production.py --preset base --steps 50000 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hclsm.config import HCLSMConfig
from hclsm.model import HCLSMWorldModel
from hclsm.training.data import SyntheticVideoDataset
from hclsm.training.trainer import Trainer
from hclsm.training.causal_env import CausalBlockWorld, CausalSceneConfig, causal_discovery_accuracy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def eval_causal(model, n_samples=20):
    if not hasattr(model, "causal_graph"):
        return {"causal_f1": 0.0, "causal_shd": 0.0}
    pred_adj = model.causal_graph.adjacency.detach().cpu()
    env = CausalBlockWorld(n_samples=n_samples, config=CausalSceneConfig(n_objects=5))
    f1, shd = 0.0, 0.0
    for i in range(n_samples):
        s = env[i]
        gt = s["causal_graph"]
        n = s["n_objects"].item()
        m = causal_discovery_accuracy(pred_adj[:n, :n], gt)
        f1 += m["f1"]
        shd += m["shd"]
    return {"causal_f1": f1 / n_samples, "causal_shd": shd / n_samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="small", choices=["tiny", "small", "base", "large"])
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--output-dir", default="./runs")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    config = getattr(HCLSMConfig, args.preset)()
    config.training.use_amp = True
    config.training.total_steps = args.steps
    config.training.warmup_steps = min(args.steps // 10, 2000)
    config.training.log_every = 100
    config.training.checkpoint_every = 5000
    config.causality.enabled = True

    if args.batch_size > 0:
        config.training.batch_size = args.batch_size
    elif args.preset in ("base", "small"):
        config.training.batch_size = 16

    run_name = args.run_name or f"hclsm_{args.preset}_{args.steps}steps"
    out = Path(args.output_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)
    config.training.checkpoint_dir = str(out / "checkpoints")
    config.to_yaml(out / "config.yaml")

    model = HCLSMWorldModel(config)
    model.enable_gradient_checkpointing()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{args.preset} | {n_params/1e6:.1f}M params | B={config.training.batch_size} | {torch.cuda.get_device_name(0)}")

    dataset = SyntheticVideoDataset(
        n_samples=max(config.training.batch_size * 1000, 50000),
        n_frames=config.perception.temporal_resolution,
        resolution=config.perception.input_resolution,
    )
    loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    trainer = Trainer(config, model, loader, device="cuda")

    logger.info(f"Training {args.steps} steps -> {out}")
    t0 = time.time()
    all_metrics = []
    best_pred = float("inf")
    step = 0

    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            metrics = trainer.train_step(batch)
            sm = {k: metrics.get(k, 0) for k in ("total", "prediction", "causal", "diversity", "hierarchy", "sigreg", "event_rate", "grad_norm", "lr")}
            sm["step"] = step
            all_metrics.append(sm)

            pred = metrics.get("prediction", float("inf"))
            if pred < best_pred:
                best_pred = pred

            if step % config.training.log_every == 0:
                elapsed = time.time() - t0
                sps = (step + 1) / elapsed if elapsed > 0 else 0
                mem = torch.cuda.max_memory_allocated() / 1024**3
                logger.info(f"Step {step:6d}/{args.steps} | total={metrics['total']:.4f} pred={metrics['prediction']:.4f} causal={metrics['causal']:.4f} div={metrics.get('diversity',0):.4f} | {sps:.1f} sps | {mem:.1f}GB")

            if step > 0 and step % 2500 == 0:
                model.eval()
                cm = eval_causal(model)
                model.train()
                logger.info(f"  Causal: F1={cm['causal_f1']:.4f} SHD={cm['causal_shd']:.1f}")
                sm.update(cm)

            if step > 0 and step % config.training.checkpoint_every == 0:
                trainer.save_checkpoint()

            if hasattr(model, "causal_graph") and config.causality.enabled:
                model.causal_graph.anneal_temperature(factor=0.9999)

            step += 1

    elapsed = time.time() - t0
    trainer.save_checkpoint()

    model.eval()
    fc = eval_causal(model, n_samples=50)
    logger.info(f"Final causal: F1={fc['causal_f1']:.4f} SHD={fc['causal_shd']:.1f}")

    with open(out / "metrics.json", "w") as f:
        json.dump(all_metrics, f)

    summary = {"preset": args.preset, "n_params": n_params, "steps": args.steps,
               "minutes": elapsed/60, "sps": args.steps/elapsed, "best_pred": best_pred,
               "final_total": all_metrics[-1]["total"], "final_pred": all_metrics[-1]["prediction"],
               "final_causal_f1": fc["causal_f1"], "final_causal_shd": fc["causal_shd"],
               "gpu_gb": torch.cuda.max_memory_allocated()/1024**3}
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Done: {elapsed/60:.1f}min | Best pred: {best_pred:.6f} | Saved: {out}")


if __name__ == "__main__":
    main()
