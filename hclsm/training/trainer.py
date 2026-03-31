"""Main training loop for HCLSM.

Sprint 2: gradient accumulation, distributed awareness (FSDP checkpoint
save/load, sampler epoch), slot attention visualizations, throughput metrics.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from hclsm.config import HCLSMConfig
from hclsm.training.schedulers import CosineWarmupScheduler
from hclsm.training.distributed import (
    save_distributed_checkpoint,
    load_distributed_checkpoint,
    is_main_process,
    get_rank,
    barrier,
)

logger = logging.getLogger(__name__)

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class Trainer:
    """Training orchestrator for HCLSMWorldModel."""

    def __init__(
        self,
        config: HCLSMConfig,
        model: nn.Module,
        train_loader: DataLoader,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config
        self.tc = config.training
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = torch.device(device)

        # Materialize any expanded/aliased parameter tensors (PyTorch 2.10+
        # nn.MultiheadAttention stores in_proj_weight as expanded view)
        params = []
        for p in model.parameters():
            if p.requires_grad:
                if not p.is_contiguous():
                    p.data = p.data.contiguous()
                params.append(p)

        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.tc.lr,
            weight_decay=self.tc.weight_decay,
        )

        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=self.tc.warmup_steps,
            total_steps=self.tc.total_steps,
        )

        # GradScaler disabled: bf16 on H100 doesn't need loss scaling.
        # Scaler with enabled=False acts as identity (no-op on all calls).
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)

        self.global_step = 0

        # Throughput tracking
        self._step_start_time: float = 0.0
        self._samples_since_log: int = 0

    # ───────────────────────────────────────────────────────────────────
    # Training step with gradient accumulation
    # ───────────────────────────────────────────────────────────────────

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Execute one training step (may span multiple micro-batches).

        Returns:
            Dict of loss values for logging.
        """
        self.model.train()
        video = batch["video"].to(self.device)
        accum_steps = self.tc.gradient_accumulation_steps
        amp_enabled = self.tc.use_amp and self.device.type == "cuda"

        B = video.shape[0]
        micro_bs = B // accum_steps if accum_steps > 1 else B

        accumulated_metrics: dict[str, float] = {}

        for micro_idx in range(accum_steps):
            # Slice micro-batch
            if accum_steps > 1:
                start = micro_idx * micro_bs
                end = start + micro_bs
                micro_video = video[start:end]
            else:
                micro_video = video

            # Extract actions if present in batch
            micro_actions = None
            if "action" in batch:
                act = batch["action"].to(self.device)
                if accum_steps > 1:
                    micro_actions = act[start:end]
                else:
                    micro_actions = act

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                output = self.model(video_frames=micro_video, actions=micro_actions, targets=micro_video)

            total_loss = output.losses["total"] / accum_steps

            # Always accumulate metrics (even if we skip backward)
            for k, v in output.losses.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                accumulated_metrics[k] = accumulated_metrics.get(k, 0.0) + val / accum_steps

            # Skip backward if loss is NaN/Inf
            if not total_loss.isfinite():
                self.optimizer.zero_grad(set_to_none=True)
                continue

            total_loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.tc.max_grad_norm,
        )

        # Step optimizer (skip if grad_norm is NaN)
        if torch.isfinite(grad_norm):
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            # EMA target encoder update — only on clean steps
            if hasattr(self.model, "update_target_encoder"):
                self.model.update_target_encoder(self.tc.ema_decay)
        else:
            self.optimizer.zero_grad(set_to_none=True)

        self.scheduler.step()
        self.global_step += 1
        self._samples_since_log += B

        # Build metrics dict
        metrics = dict(accumulated_metrics)
        metrics["grad_norm"] = (
            grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        )
        metrics["lr"] = self.scheduler.get_last_lr()[0]

        return metrics

    # ───────────────────────────────────────────────────────────────────
    # Main training loop
    # ───────────────────────────────────────────────────────────────────

    def train(self, n_epochs: int = 1) -> None:
        """Run the full training loop."""
        for epoch in range(n_epochs):
            # Set epoch on DistributedSampler for proper shuffling
            if isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)

            self._step_start_time = time.monotonic()
            self._samples_since_log = 0

            for batch in self.train_loader:
                metrics = self.train_step(batch)

                if self.global_step % self.tc.log_every == 0:
                    self._log_metrics(metrics)
                    self._step_start_time = time.monotonic()
                    self._samples_since_log = 0

                if (
                    self.global_step % self.tc.vis_every == 0
                    and is_main_process()
                ):
                    self._log_visualizations(batch)

                if self.global_step % self.tc.checkpoint_every == 0:
                    self.save_checkpoint()

                if self.global_step >= self.tc.total_steps:
                    logger.info("Reached total_steps, stopping.")
                    return

    # ───────────────────────────────────────────────────────────────────
    # Logging
    # ───────────────────────────────────────────────────────────────────

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics to console and wandb (with throughput)."""
        # Compute throughput
        elapsed = time.monotonic() - self._step_start_time
        if elapsed > 0 and self._samples_since_log > 0:
            samples_per_sec = self._samples_since_log / elapsed
            metrics["throughput/samples_per_sec"] = samples_per_sec
            # Tokens ~ frames * patches_per_frame
            pc = self.config.perception
            patches_per_frame = (pc.input_resolution // pc.patch_size) ** 2
            tokens_per_sample = pc.temporal_resolution * patches_per_frame
            metrics["throughput/tokens_per_sec"] = samples_per_sec * tokens_per_sample

        if is_main_process():
            loss_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            logger.info(f"Step {self.global_step} | {loss_str}")

        if WANDB_AVAILABLE and wandb.run is not None and is_main_process():
            wandb.log(
                {f"train/{k}": v for k, v in metrics.items()},
                step=self.global_step,
            )

    def _log_visualizations(self, batch: dict[str, torch.Tensor]) -> None:
        """Log slot attention visualizations to wandb."""
        if not (WANDB_AVAILABLE and wandb.run is not None):
            return

        try:
            self.model.eval()
            video = batch["video"].to(self.device)

            # Use first sample only
            with torch.no_grad():
                output = self.model(
                    video_frames=video[:1], targets=video[:1], return_attention=True,
                )

            vis_dict: dict = {}

            # Slot attention maps: (1, T, N_max, M)
            if output.slot_attention_maps is not None:
                attn = output.slot_attention_maps[0]  # (T, N, M)
                # Take first frame, reshape to spatial
                t0_attn = attn[0]  # (N, M)
                pc = self.config.perception
                grid_h = pc.input_resolution // pc.patch_size
                n_slots_to_show = min(t0_attn.shape[0], 8)
                for s in range(n_slots_to_show):
                    attn_map = t0_attn[s, :grid_h * grid_h].reshape(grid_h, grid_h)
                    attn_np = attn_map.cpu().float().numpy()
                    vis_dict[f"slots/attention_slot_{s}"] = wandb.Image(
                        attn_np, caption=f"Slot {s} attention (frame 0)",
                    )

            # Alive mask: (1, T, N)
            if output.alive_mask is not None:
                alive = output.alive_mask[0]  # (T, N)
                n_alive_per_frame = (alive > 0.5).float().sum(dim=1)  # (T,)
                vis_dict["slots/n_alive_per_frame"] = wandb.Histogram(
                    n_alive_per_frame.cpu().numpy(),
                )

            # Event scores: (1, T)
            if output.event_scores is not None:
                scores = output.event_scores[0].cpu().numpy()
                vis_dict["dynamics/event_scores"] = wandb.Histogram(scores)

            if vis_dict:
                wandb.log(vis_dict, step=self.global_step)

        except Exception as e:
            logger.warning(f"Visualization failed at step {self.global_step}: {e}")
        finally:
            self.model.train()

    # ───────────────────────────────────────────────────────────────────
    # Checkpointing (FSDP-aware)
    # ───────────────────────────────────────────────────────────────────

    def save_checkpoint(self, path: str | None = None) -> None:
        """Save training checkpoint (FSDP-aware)."""
        if path is not None:
            ckpt_dir = str(Path(path).parent)
        else:
            ckpt_dir = self.tc.checkpoint_dir

        saved = save_distributed_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            global_step=self.global_step,
            checkpoint_dir=ckpt_dir,
            rank=get_rank(),
        )
        if saved:
            logger.info(f"Saved checkpoint to {saved}")

        barrier()

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint (FSDP-aware)."""
        self.global_step = load_distributed_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            path=path,
        )
        logger.info(f"Loaded checkpoint from {path} at step {self.global_step}")
