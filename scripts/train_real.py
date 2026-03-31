"""Production training on real video data with multi-GPU FSDP.

Single GPU:
    python scripts/train_real.py --preset base --data-dir /path/to/UCF-101

Multi-GPU (4x H100):
    torchrun --nproc_per_node=4 scripts/train_real.py --preset base --data-dir /path/to/UCF-101
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hclsm.config import HCLSMConfig
from hclsm.model import HCLSMWorldModel
from hclsm.training.data import build_dataloader
from hclsm.training.distributed import (
    setup_distributed,
    teardown_distributed,
    wrap_model_distributed,
    is_main_process,
    get_rank,
    get_world_size,
)
from hclsm.training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="base", choices=["tiny", "small", "base", "large"])
    parser.add_argument("--data-dir", default="", help="Path to video dir or HF repo_id")
    parser.add_argument("--dataset", default="openx", choices=["video", "openx", "synthetic"])
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=0, help="Per-GPU batch size (0=auto)")
    parser.add_argument("--lr", type=float, default=0, help="Learning rate (0=use preset)")
    parser.add_argument("--output-dir", default="./runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    args = parser.parse_args()

    # ── Distributed ──
    rank, world_size = setup_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # ── Config ──
    config = getattr(HCLSMConfig, args.preset)()
    config.training.use_amp = True
    config.training.total_steps = args.steps
    config.training.warmup_steps = min(args.steps // 10, 2000)
    config.training.log_every = 50
    config.training.checkpoint_every = 2500
    config.causality.enabled = True
    config.training.dataset_name = args.dataset
    config.training.data_dir = args.data_dir
    # d_action will be set after dataset is loaded (varies by dataset)
    config.training.use_gradient_checkpointing = False  # Disabled: causes NaN with real data

    # Use default small config — no overrides (proven to work on real data)

    # Per-GPU batch size
    if args.batch_size > 0:
        config.training.batch_size = args.batch_size
    else:
        # Auto: fit in GPU memory
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9 if torch.cuda.is_available() else 16
        if args.preset == "base":
            config.training.batch_size = 4 if gpu_mem < 40 else 8
        elif args.preset == "large":
            config.training.batch_size = 2 if gpu_mem < 40 else 4
        else:
            config.training.batch_size = 8

    if args.lr > 0:
        config.training.lr = args.lr

    # DDP for multi-GPU (FSDP only needed for models too large for one GPU)
    if world_size > 1:
        config.training.fsdp_enabled = False  # Use DDP instead

    # Effective batch size
    eff_batch = config.training.batch_size * world_size
    if is_main_process():
        logger.info(f"Config: {args.preset} | LR={config.training.lr} | B={config.training.batch_size}x{world_size}={eff_batch}")
        logger.info(f"Data: {args.data_dir} | Steps: {args.steps}")

    # ── Output ──
    run_name = args.run_name or f"hclsm_{args.preset}_{world_size}gpu_{args.steps}steps"
    out = Path(args.output_dir) / run_name
    if is_main_process():
        out.mkdir(parents=True, exist_ok=True)
        config.training.checkpoint_dir = str(out / "checkpoints")
        config.to_yaml(out / "config.yaml")

    # ── Data (load first to detect d_action) ──
    loader = build_dataloader(
        config, split="train",
        distributed=(world_size > 1),
        world_size=world_size,
        rank=rank,
    )
    if is_main_process():
        logger.info(f"Dataset: {len(loader.dataset)} videos, {len(loader)} batches/epoch")

    # Auto-detect d_action from dataset
    ds = loader.dataset
    if hasattr(ds, "d_action") and ds.d_action > 0:
        config.dynamics.level2.d_action = ds.d_action
        if is_main_process():
            logger.info(f"Auto-detected d_action={ds.d_action}")

    # ── FSDP wrap class registration ──
    from hclsm.training.distributed import register_hclsm_wrap_classes
    register_hclsm_wrap_classes()

    # ── Model ──
    model = HCLSMWorldModel(config)
    if config.training.use_gradient_checkpointing:
        model.enable_gradient_checkpointing()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process():
        logger.info(f"Model: {n_params/1e6:.1f}M params")

    # Distributed wrap (DDP or FSDP)
    model = model.to(device)
    if world_size > 1:
        if config.training.fsdp_enabled:
            model = wrap_model_distributed(model, config)
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True,
            )
            if is_main_process():
                logger.info(f"DDP wrapped on {world_size} GPUs")

    # ── Trainer ──
    trainer = Trainer(config, model, loader, device=device)

    if args.resume:
        trainer.load_checkpoint(args.resume)
        if is_main_process():
            logger.info(f"Resumed from {args.resume}")

    # ── Train ──
    if is_main_process():
        logger.info("Starting training...")

    t0 = time.time()
    all_metrics = []
    step = 0
    epoch = 0

    try:
        while step < args.steps:
            epoch += 1
            if hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(epoch)

            for batch in loader:
                if step >= args.steps:
                    break

                # ── Two-stage training schedule ──
                # Stage 1 (0 - 40%): Reconstruction-only → force slot decomposition
                # Stage 2 (40% - 100%): Full JEPA prediction + SBD regularizer
                stage1_end = int(args.steps * 0.4)
                stage = 1 if step < stage1_end else 2

                # Set stage on model (accessed in forward pass)
                unwrapped = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
                unwrapped._training_stage = stage

                metrics = trainer.train_step(batch)
                step_m = {k: metrics.get(k, 0) for k in ("total", "prediction", "causal", "diversity", "tracking", "spatial_recon", "grad_norm", "lr")}
                step_m["step"] = step
                step_m["stage"] = stage
                all_metrics.append(step_m)

                if step % config.training.log_every == 0 and is_main_process():
                    elapsed = time.time() - t0
                    sps = (step + 1) / elapsed if elapsed > 0 else 0
                    mem = torch.cuda.max_memory_allocated() / 1024**3
                    gn = metrics.get("grad_norm", 0)
                    nan_grad = "NaN" if (isinstance(gn, float) and gn != gn) else f"{gn:.2f}"
                    sbd = metrics.get("spatial_recon", 0)
                    logger.info(
                        f"[S{stage}] Step {step:6d}/{args.steps} | "
                        f"total={metrics.get('total',0):.4f} pred={metrics.get('prediction',0):.4f} "
                        f"sbd={sbd:.4f} grad={nan_grad} | "
                        f"{sps:.2f} sps | {mem:.1f}GB"
                    )

                if step > 0 and step % config.training.checkpoint_every == 0:
                    trainer.save_checkpoint()

                if hasattr(model, "causal_graph") and config.causality.enabled:
                    model.causal_graph.anneal_temperature(factor=0.9999)

                step += 1

    except KeyboardInterrupt:
        if is_main_process():
            logger.info("Interrupted")
    except Exception as e:
        if is_main_process():
            logger.error(f"Training error: {e}")
    finally:
        try:
            if is_main_process():
                trainer.save_checkpoint()
        except Exception:
            pass  # Don't crash on checkpoint save failure
        teardown_distributed()

    if is_main_process():
        elapsed = time.time() - t0
        with open(out / "metrics.json", "w") as f:
            json.dump(all_metrics, f)

        summary = {
            "preset": args.preset, "n_params": n_params, "steps": step,
            "world_size": world_size, "batch_per_gpu": config.training.batch_size,
            "eff_batch": eff_batch, "lr": config.training.lr,
            "minutes": elapsed / 60, "sps": step / max(elapsed, 1),
            "final_total": all_metrics[-1]["total"] if all_metrics else 0,
            "final_pred": all_metrics[-1]["prediction"] if all_metrics else 0,
            "gpu_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }
        with open(out / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Done: {elapsed/60:.1f}min | {step} steps | Saved: {out}")


if __name__ == "__main__":
    main()
