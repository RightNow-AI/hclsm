"""Distributed training utilities for HCLSM.

Sprint 2: Full FSDP wrapping, process group setup/teardown, and
FSDP-aware checkpoint save/load.
"""

from __future__ import annotations

import logging
import os
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Guard imports — FSDP requires PyTorch >= 2.0
try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
        CPUOffload,
    )
    from torch.distributed.fsdp.wrap import (
        size_based_auto_wrap_policy,
        transformer_auto_wrap_policy,
    )
    from torch.distributed.fsdp import (
        FullStateDictConfig,
        StateDictType,
    )
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# Process group setup / teardown
# ═══════════════════════════════════════════════════════════════════════════

def setup_distributed(backend: str = "nccl") -> tuple[int, int]:
    """Initialize the distributed process group.

    Reads RANK, WORLD_SIZE, LOCAL_RANK from environment (set by torchrun).

    Returns:
        (rank, world_size) tuple.
    """
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size <= 1:
        logger.info("Single-process mode, skipping dist init.")
        return rank, world_size

    if dist.is_initialized():
        logger.info("Process group already initialized.")
        return rank, world_size

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    logger.info(f"Distributed init: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    return rank, world_size


def teardown_distributed() -> None:
    """Destroy the distributed process group if active."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Destroyed process group.")


def is_main_process() -> bool:
    """True on rank 0 or when not using distributed."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def barrier() -> None:
    """Synchronize all processes (no-op if not distributed)."""
    if dist.is_initialized():
        dist.barrier()


# ═══════════════════════════════════════════════════════════════════════════
# FSDP wrapping
# ═══════════════════════════════════════════════════════════════════════════

# Modules that should be individually wrapped as FSDP units.
# This list mirrors the architecture layers so each one can be sharded.
_TRANSFORMER_WRAP_CLASSES: set[type] = set()


def register_fsdp_wrap_class(cls: type) -> type:
    """Register a module class for FSDP transformer_auto_wrap_policy."""
    _TRANSFORMER_WRAP_CLASSES.add(cls)
    return cls


def register_hclsm_wrap_classes() -> None:
    """Register all HCLSM modules for optimal FSDP sharding.

    Each registered class becomes an individually-sharded FSDP unit,
    giving clean boundaries for parameter distribution across GPUs.
    """
    from hclsm.perception.vision_encoder import VisionEncoder
    from hclsm.objects.dynamic_slots import DynamicSlotAttention
    from hclsm.objects.relation_graph import RelationGraph
    from hclsm.dynamics.level0_ssm import Level0SSM
    from hclsm.dynamics.level1_event import Level1EventDynamics
    from hclsm.dynamics.level2_goal import Level2GoalDynamics

    for cls in [VisionEncoder, DynamicSlotAttention, RelationGraph,
                Level0SSM, Level1EventDynamics, Level2GoalDynamics]:
        register_fsdp_wrap_class(cls)


def _get_sharding_strategy(name: str) -> Any:
    """Map config string to ShardingStrategy enum."""
    if not FSDP_AVAILABLE:
        raise RuntimeError("FSDP not available in this PyTorch build")
    mapping = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }
    if name not in mapping:
        raise ValueError(f"Unknown sharding strategy '{name}', choose from {list(mapping)}")
    return mapping[name]


def _build_mixed_precision(use_amp: bool) -> Any | None:
    """Build FSDP MixedPrecision policy."""
    if not use_amp or not FSDP_AVAILABLE:
        return None
    return MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )


def wrap_model_fsdp(
    model: nn.Module,
    sharding_strategy: str = "FULL_SHARD",
    cpu_offload: bool = False,
    use_amp: bool = True,
    min_num_params: int = 1_000_000,
) -> nn.Module:
    """Wrap a model with FSDP.

    Args:
        model: The model to wrap.
        sharding_strategy: One of FULL_SHARD, SHARD_GRAD_OP, NO_SHARD, HYBRID_SHARD.
        cpu_offload: Offload parameters to CPU when not in use.
        use_amp: Enable mixed-precision inside FSDP.
        min_num_params: Minimum parameter count for auto-wrapping sub-modules.

    Returns:
        FSDP-wrapped model.
    """
    if not FSDP_AVAILABLE:
        raise RuntimeError(
            "FSDP requires PyTorch >= 2.0. Please upgrade or disable fsdp_enabled."
        )

    # Pick wrapping policy: transformer-class-based if any registered,
    # otherwise size-based fallback
    if _TRANSFORMER_WRAP_CLASSES:
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=_TRANSFORMER_WRAP_CLASSES,
        )
    else:
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params,
        )

    strategy = _get_sharding_strategy(sharding_strategy)
    mp_policy = _build_mixed_precision(use_amp)
    offload = CPUOffload(offload_params=True) if cpu_offload else None

    wrapped = FSDP(
        model,
        sharding_strategy=strategy,
        cpu_offload=offload,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,  # Required when modules mix frozen/unfrozen params
    )
    logger.info(
        f"FSDP wrapped: strategy={sharding_strategy}, "
        f"cpu_offload={cpu_offload}, amp={use_amp}"
    )
    return wrapped


# ═══════════════════════════════════════════════════════════════════════════
# Unified wrapper (routes to FSDP or no-op)
# ═══════════════════════════════════════════════════════════════════════════

def wrap_model_distributed(model: nn.Module, config: Any | None = None) -> nn.Module:
    """Wrap model for distributed training based on config.

    If ``config.training.fsdp_enabled`` is True and distributed is active,
    wraps with FSDP. Otherwise returns model unchanged.

    Args:
        model: The model to potentially wrap.
        config: HCLSMConfig. If None, returns model unchanged (Sprint 1 compat).

    Returns:
        Wrapped (or unwrapped) model.
    """
    if config is None:
        return model

    tc = config.training
    if not tc.fsdp_enabled:
        return model

    if not dist.is_initialized() or get_world_size() <= 1:
        logger.info("FSDP enabled but not in distributed mode, skipping wrap.")
        return model

    return wrap_model_fsdp(
        model,
        sharding_strategy=tc.fsdp_sharding_strategy,
        cpu_offload=tc.fsdp_cpu_offload,
        use_amp=tc.use_amp,
    )


# ═══════════════════════════════════════════════════════════════════════════
# FSDP-aware checkpointing
# ═══════════════════════════════════════════════════════════════════════════

def save_distributed_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    global_step: int,
    checkpoint_dir: str,
    rank: int = 0,
) -> str | None:
    """Save a checkpoint that works for both FSDP and non-FSDP models.

    For FSDP models, gathers the full state dict on rank 0.
    Only rank 0 writes to disk.

    Returns:
        Path to saved checkpoint (rank 0) or None (other ranks).
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{global_step}.pt"

    is_fsdp = FSDP_AVAILABLE and isinstance(model, FSDP)

    if is_fsdp:
        # Gather full state dict on rank 0
        full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
            model_state = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer)
    else:
        model_state = model.state_dict()
        optim_state = optimizer.state_dict()

    if rank == 0 or not dist.is_initialized():
        checkpoint = {
            "model": model_state,
            "optimizer": optim_state,
            "scheduler": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
            "scaler": scaler.state_dict() if hasattr(scaler, "state_dict") else None,
            "global_step": global_step,
            "is_fsdp": is_fsdp,
        }
        torch.save(checkpoint, str(path))
        logger.info(f"Saved checkpoint to {path}")
        return str(path)

    return None


def load_distributed_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    path: str,
) -> int:
    """Load a checkpoint for both FSDP and non-FSDP models.

    Args:
        model: The (possibly FSDP-wrapped) model.
        optimizer: Optimizer.
        scheduler: LR scheduler.
        scaler: GradScaler.
        path: Path to checkpoint file.

    Returns:
        global_step from the checkpoint.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    is_fsdp = FSDP_AVAILABLE and isinstance(model, FSDP)

    if is_fsdp:
        full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
            model.load_state_dict(ckpt["model"])
            if ckpt.get("optimizer") is not None:
                optim_state = FSDP.optim_state_dict_to_load(
                    model, optimizer, ckpt["optimizer"],
                )
                optimizer.load_state_dict(optim_state)
    else:
        model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])

    if ckpt.get("scheduler") is not None and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler"])

    if ckpt.get("scaler") is not None and hasattr(scaler, "load_state_dict"):
        scaler.load_state_dict(ckpt["scaler"])

    global_step = ckpt.get("global_step", 0)
    logger.info(f"Loaded checkpoint from {path} at step {global_step}")
    return global_step
