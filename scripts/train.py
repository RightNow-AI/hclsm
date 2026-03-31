"""Launch HCLSM training.

Sprint 2: supports distributed training (torchrun), gradient accumulation,
gradient checkpointing, build_dataloader(), and wandb logging.
"""

from __future__ import annotations

import argparse
import logging

import torch

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HCLSM world model")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument(
        "--preset", type=str, default="tiny",
        choices=["tiny", "small", "base", "large"],
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="hclsm")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    # ── Distributed setup ──
    rank, world_size = setup_distributed()
    distributed = world_size > 1
    if distributed:
        device = torch.device("cuda", int(__import__("os").environ.get("LOCAL_RANK", 0)))
    else:
        device = torch.device(args.device)

    # ── Config ──
    if args.config:
        config = HCLSMConfig.from_yaml(args.config)
    else:
        config = getattr(HCLSMConfig, args.preset)()

    # ── Wandb ──
    if WANDB_AVAILABLE and not args.no_wandb and is_main_process():
        wandb.init(
            project=args.wandb_project,
            config=config.to_dict(),
            name=f"hclsm-{args.preset}",
        )

    # ── Model ──
    model = HCLSMWorldModel(config)

    if config.training.use_gradient_checkpointing:
        model.enable_gradient_checkpointing()
        if is_main_process():
            logger.info("Gradient checkpointing enabled")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process():
        logger.info(f"Model created: {n_params / 1e6:.1f}M trainable parameters")

    model = wrap_model_distributed(model, config)

    # ── Data ──
    loader = build_dataloader(
        config, split="train",
        distributed=distributed,
        world_size=world_size,
        rank=rank,
    )

    # ── Trainer ──
    trainer = Trainer(config, model, loader, device=device)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    try:
        trainer.train(n_epochs=args.epochs)
    finally:
        teardown_distributed()
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
