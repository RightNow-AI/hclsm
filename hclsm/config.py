"""HCLSM configuration system.

All hyperparameters are defined as nested dataclasses with preset configurations
for tiny (50M), small (200M), base (800M), and large (3B) model variants.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PerceptionConfig:
    """Vision encoder configuration."""

    d_model: int = 768
    n_layers: int = 24
    n_heads: int = 12
    patch_size: int = 16
    input_resolution: int = 224
    temporal_resolution: int = 16
    dropout: float = 0.0
    init_from: str | None = None


@dataclass
class ObjectConfig:
    """Object decomposition (slot attention + GNN) configuration."""

    d_slot: int = 256
    n_max_slots: int = 64
    n_iterations: int = 7
    existence_threshold: float = 0.5
    d_edge: int = 128
    gnn_rounds: int = 2
    birth_threshold: float = 0.7


@dataclass
class Level0Config:
    """Level 0 SSM (fast physics) configuration."""

    d_state: int = 64
    n_blocks: int = 4
    expand_ratio: int = 2


@dataclass
class Level1Config:
    """Level 1 event dynamics (sparse transformer) configuration."""

    d_model: int = 512
    n_layers: int = 4
    n_heads: int = 8


@dataclass
class Level2Config:
    """Level 2 goal dynamics (compressed transformer) configuration."""

    d_model: int = 768
    n_layers: int = 6
    n_heads: int = 12
    n_summary_tokens: int = 8
    context_window: int = 256
    d_goal: int = 0
    d_action: int = 0


@dataclass
class DynamicsConfig:
    """Hierarchical dynamics engine configuration."""

    level0: Level0Config = field(default_factory=Level0Config)
    level1: Level1Config = field(default_factory=Level1Config)
    level2: Level2Config = field(default_factory=Level2Config)
    event_threshold: float = 0.7
    event_window_size: int = 8
    min_events_for_l2: int = 4


@dataclass
class CausalityConfig:
    """Causal reasoning module configuration."""

    enabled: bool = True
    dag_penalty_rho_init: float = 0.1
    dag_penalty_rho_max: float = 10.0
    sparsity_lambda: float = 0.01
    temperature_init: float = 1.0
    temperature_min: float = 0.1


@dataclass
class MemoryConfig:
    """Continual learning memory configuration."""

    episodic_size: int = 10000
    d_memory: int = 256
    consolidation_every: int = 1000


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    batch_size: int = 256
    lr: float = 3e-4
    warmup_steps: int = 10000
    total_steps: int = 500_000
    weight_decay: float = 0.05
    max_grad_norm: float = 1.0
    ema_decay: float = 0.996
    use_amp: bool = True

    # Loss weights
    lambda_pred: float = 1.0
    lambda_obj: float = 0.5
    lambda_aux: float = 0.1
    lambda_causal: float = 0.1
    lambda_hierarchy: float = 0.25
    lambda_sigreg: float = 0.1

    # Distributed training
    gradient_accumulation_steps: int = 1
    num_workers: int = 4
    distributed_backend: str = "nccl"
    seed: int = 42

    # FSDP
    fsdp_enabled: bool = False
    fsdp_sharding_strategy: str = "FULL_SHARD"
    fsdp_cpu_offload: bool = False

    # Data pipeline
    data_dir: str = ""
    dataset_name: str = "synthetic"
    video_backend: str = "decord"

    # Memory optimization
    use_gradient_checkpointing: bool = False

    # Logging & visualization
    log_every: int = 100
    vis_every: int = 500
    checkpoint_every: int = 5000
    checkpoint_dir: str = "checkpoints"


@dataclass
class HCLSMConfig:
    """Top-level HCLSM configuration.

    Contains all hyperparameters for the complete architecture.
    Use class methods tiny(), small(), base(), large() for preset configs.
    """

    d_world: int = 768
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    objects: ObjectConfig = field(default_factory=ObjectConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    causality: CausalityConfig = field(default_factory=CausalityConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check configuration constraints."""
        assert self.perception.d_model % self.perception.n_heads == 0, (
            f"d_model ({self.perception.d_model}) must be divisible by "
            f"n_heads ({self.perception.n_heads})"
        )
        assert self.perception.input_resolution % self.perception.patch_size == 0, (
            f"input_resolution ({self.perception.input_resolution}) must be "
            f"divisible by patch_size ({self.perception.patch_size})"
        )
        assert self.dynamics.level1.d_model % self.dynamics.level1.n_heads == 0, (
            f"Level1 d_model ({self.dynamics.level1.d_model}) must be divisible "
            f"by n_heads ({self.dynamics.level1.n_heads})"
        )
        assert self.dynamics.level2.d_model % self.dynamics.level2.n_heads == 0, (
            f"Level2 d_model ({self.dynamics.level2.d_model}) must be divisible "
            f"by n_heads ({self.dynamics.level2.n_heads})"
        )

    @classmethod
    def tiny(cls) -> HCLSMConfig:
        """50M parameter config for debugging."""
        return cls(
            d_world=256,
            perception=PerceptionConfig(
                d_model=384, n_layers=6, n_heads=6,
            ),
            objects=ObjectConfig(
                d_slot=128, n_max_slots=16, n_iterations=3,
                d_edge=64, gnn_rounds=1,
            ),
            dynamics=DynamicsConfig(
                level0=Level0Config(d_state=32, n_blocks=2),
                level1=Level1Config(d_model=256, n_layers=2, n_heads=4),
                level2=Level2Config(d_model=384, n_layers=2, n_heads=6),
            ),
            causality=CausalityConfig(enabled=False),
            memory=MemoryConfig(episodic_size=1000, d_memory=128),
            training=TrainingConfig(batch_size=64),
        )

    @classmethod
    def small(cls) -> HCLSMConfig:
        """200M parameter config for research."""
        return cls(
            d_world=384,
            perception=PerceptionConfig(
                d_model=512, n_layers=12, n_heads=8,
            ),
            objects=ObjectConfig(
                d_slot=192, n_max_slots=32, n_iterations=5,
                d_edge=96, gnn_rounds=2,
            ),
            dynamics=DynamicsConfig(
                level0=Level0Config(d_state=48, n_blocks=3),
                level1=Level1Config(d_model=384, n_layers=3, n_heads=6),
                level2=Level2Config(d_model=512, n_layers=3, n_heads=8),
            ),
            causality=CausalityConfig(enabled=True),
            memory=MemoryConfig(episodic_size=5000, d_memory=192),
            training=TrainingConfig(batch_size=128, lr=1.5e-4),
        )

    @classmethod
    def base(cls) -> HCLSMConfig:
        """800M parameter config for benchmarks."""
        return cls(
            d_world=768,
            perception=PerceptionConfig(
                d_model=768, n_layers=24, n_heads=12,
                init_from="v-jepa-2-base",
            ),
            objects=ObjectConfig(
                d_slot=256, n_max_slots=64, n_iterations=7,
                d_edge=128, gnn_rounds=2,
            ),
            dynamics=DynamicsConfig(
                level0=Level0Config(d_state=64, n_blocks=4),
                level1=Level1Config(d_model=512, n_layers=4, n_heads=8),
                level2=Level2Config(d_model=768, n_layers=6, n_heads=12),
            ),
            causality=CausalityConfig(enabled=True),
            memory=MemoryConfig(episodic_size=10000, d_memory=256),
            training=TrainingConfig(batch_size=2, lr=1e-4, max_grad_norm=0.5),  # B=2 max for base (NaN at B>=4)
        )

    @classmethod
    def large(cls) -> HCLSMConfig:
        """3B parameter config, flagship."""
        return cls(
            d_world=1024,
            perception=PerceptionConfig(
                d_model=1024, n_layers=32, n_heads=16,
                init_from="v-jepa-2-large",
            ),
            objects=ObjectConfig(
                d_slot=384, n_max_slots=128, n_iterations=7,
                d_edge=192, gnn_rounds=3,
            ),
            dynamics=DynamicsConfig(
                level0=Level0Config(d_state=96, n_blocks=6),
                level1=Level1Config(d_model=768, n_layers=6, n_heads=12),
                level2=Level2Config(d_model=1024, n_layers=8, n_heads=16),
            ),
            causality=CausalityConfig(enabled=True),
            memory=MemoryConfig(episodic_size=50000, d_memory=384,
                                consolidation_every=500),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to nested dictionary."""
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> HCLSMConfig:
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HCLSMConfig:
        """Construct from nested dictionary, recursively building sub-configs."""
        return cls(
            d_world=data.get("d_world", 768),
            perception=_build_dataclass(
                PerceptionConfig, data.get("perception", {})
            ),
            objects=_build_dataclass(ObjectConfig, data.get("objects", {})),
            dynamics=_build_dynamics(data.get("dynamics", {})),
            causality=_build_dataclass(
                CausalityConfig, data.get("causality", {})
            ),
            memory=_build_dataclass(MemoryConfig, data.get("memory", {})),
            training=_build_dataclass(TrainingConfig, data.get("training", {})),
        )


def _build_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Build a dataclass from a dict, ignoring unknown keys."""
    valid_keys = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return cls(**filtered)


def _build_dynamics(data: dict[str, Any]) -> DynamicsConfig:
    """Build DynamicsConfig with nested level configs."""
    level0 = _build_dataclass(Level0Config, data.get("level0", {}))
    level1 = _build_dataclass(Level1Config, data.get("level1", {}))
    level2 = _build_dataclass(Level2Config, data.get("level2", {}))
    top_level = {
        k: v for k, v in data.items()
        if k not in ("level0", "level1", "level2")
        and k in {f.name for f in fields(DynamicsConfig)}
    }
    return DynamicsConfig(level0=level0, level1=level1, level2=level2, **top_level)
