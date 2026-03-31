# HCLSM: Hierarchical Causal Latent State Machines

**A world model that sees objects, tracks their dynamics at multiple timescales, and learns how they interact.**

[![arXiv](https://img.shields.io/badge/arXiv-2603.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2603.xxxxx)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-171%20passing-brightgreen.svg)]()

> [RightNow AI](https://www.rightnowai.co/)

---

## What is HCLSM?

Current world models predict the future as a flat vector. They cannot tell you which object moved, why it moved, or what would happen if you pushed it differently. HCLSM changes this.

HCLSM decomposes video into **object slots**, models their dynamics through a **three-level temporal hierarchy** (SSM for physics, sparse transformer for events, goal transformer for planning), and learns **causal interaction structure** through a graph neural network. A spatial broadcast decoder forces each slot to own a distinct region of the image, producing unsupervised object segmentation without any labels.

The architecture is trained in two stages. Stage 1 trains with reconstruction only, forcing slots to specialize into objects. Stage 2 activates JEPA-style latent prediction on top of the decomposed representation.

## Architecture

```
Video (B, T, C, H, W)
    |
    v
[Layer 1] Vision Transformer Encoder
    |
    v
[Layer 2] Dynamic Slot Attention + Spatial Broadcast Decoder + GNN
    |          Each slot reconstructs its own spatial region.
    |          Softmax over slots creates object-level competition.
    v
[Layer 3] Hierarchical Dynamics Engine
    |--- Level 0: Selective SSM (per-object continuous physics)
    |--- Level 1: Sparse Transformer (fires at detected events only)
    |--- Level 2: Goal Transformer (compressed abstract reasoning)
    v
[Layer 4] Causal Graph (DAG-constrained adjacency + GNN edge weights)
    |
    v
[Layer 5] Episodic Memory (Hopfield retrieval + EWC consolidation)
```

## Key Results

Trained on the PushT robotic manipulation task from Open X-Embodiment (real robot data, not simulation).

| Metric | Value |
|--------|-------|
| Prediction MSE | 0.008 |
| SBD reconstruction | 0.008 |
| Tracking loss | 0.016 |
| SSM kernel speedup | 38x (Triton vs PyTorch) |
| Model size | 68M parameters |
| Training | 50K steps on NVIDIA H100 |

## Installation

```bash
git clone https://github.com/rightnow-ai/hclsm.git
cd hclsm
pip install -e .
```

Optional dependencies:

```bash
pip install -e ".[kernels]"  # Triton + Mamba + FlashAttention
pip install -e ".[video]"    # decord + webdataset for video loading
pip install -e ".[dev]"      # pytest + ruff + mypy
```

## Quick Start

### Train on synthetic data

```python
from hclsm.config import HCLSMConfig
from hclsm.model import HCLSMWorldModel
from hclsm.training.data import SyntheticVideoDataset
from hclsm.training.trainer import Trainer
from torch.utils.data import DataLoader

config = HCLSMConfig.tiny()
model = HCLSMWorldModel(config)

dataset = SyntheticVideoDataset(n_samples=1000, n_frames=16, resolution=224)
loader = DataLoader(dataset, batch_size=8, shuffle=True)
trainer = Trainer(config, model, loader, device="cuda")

for step, batch in enumerate(loader):
    metrics = trainer.train_step(batch)
    if step % 100 == 0:
        print(f"Step {step}: loss={metrics['total']:.4f} pred={metrics['prediction']:.4f}")
```

### Train on real robot data (Open X-Embodiment)

```bash
pip install lerobot

python scripts/train_real.py \
    --preset small \
    --dataset openx \
    --data-dir lerobot/pusht \
    --steps 50000 \
    --batch-size 4
```

### Evaluate a trained checkpoint

```bash
python scripts/eval_full.py \
    --checkpoint runs/your_run/checkpoints/step_50000.pt \
    --preset small \
    --data-dir lerobot/pusht \
    --output-dir eval_results
```

This produces: slot attention heatmaps, spatial decomposition maps, causal graph visualization, event detection timelines, prediction horizon curves, PCA trajectories, and a LaTeX metrics table.

## Model Configurations

| Config | Parameters | d_world | Slots | ViT Layers | SSM Blocks |
|--------|-----------|---------|-------|------------|------------|
| `tiny` | 22M | 256 | 16 | 6 | 2 |
| `small` | 68M | 384 | 32 | 12 | 3 |
| `base` | 262M | 768 | 64 | 24 | 4 |
| `large` | 3B | 1024 | 128 | 32 | 6 |

```python
config = HCLSMConfig.tiny()   # debugging
config = HCLSMConfig.small()  # research (trained and evaluated)
config = HCLSMConfig.base()   # benchmarks (requires bf16 stability fixes)
config = HCLSMConfig.large()  # flagship (not yet trained)
```

## Project Structure

```
hclsm/
  config.py                 # Nested dataclass configs with 4 presets
  model.py                  # Full 5-layer architecture + two-stage training
  perception/               # ViT encoder, multimodal fuser, tokenizer
  objects/                  # Slot attention, spatial broadcast decoder, GNN, tracking
  dynamics/                 # Level 0 SSM, Level 1 event transformer, Level 2 goal transformer
  causality/                # Causal graph (NOTEARS DAG), counterfactual simulator
  memory/                   # Episodic memory (Hopfield), EWC consolidation
  training/                 # Trainer, losses, data pipeline, FSDP, benchmarks
  inference/                # World simulator, CEM planner, MPPI planner
  kernels/                  # Triton SSM scan, fused slot attention, sparse event attention
scripts/
  train_real.py             # Production training with two-stage schedule
  eval_full.py              # Full evaluation suite (8 analyses)
  benchmark_kernels.py      # Kernel performance benchmarks
configs/
  hclsm_tiny.yaml           # Preset configs
  ablations/                # 7 ablation configs (no_objects, no_hierarchy, etc.)
tests/                      # 171 tests across 13 files
```

## Custom Triton Kernels

The SSM scan is the computational bottleneck when processing object tracks sequentially. Our Triton kernel parallelizes across batch and feature dimensions:

| Config | Sequential (PyTorch) | Triton | Speedup |
|--------|---------------------|--------|---------|
| Tiny (128 tracks) | 6.22 ms | 0.16 ms | 39x |
| Base (512 tracks) | 69.64 ms | 1.83 ms | 38x |

All kernels auto-dispatch: Triton on GPU, PyTorch fallback on CPU.

```python
from hclsm.kernels import ssm_scan_fn, slot_attention_fn

# Automatically uses Triton on CUDA, sequential on CPU
y = ssm_scan_fn(x, dt, A_log, B, C)
out, attn = slot_attention_fn(Q, K, V)
```

## Two-Stage Training

HCLSM trains in two stages (following SlotFormer and DINOSAUR):

**Stage 1 (0-40% of training):** Reconstruction only. The spatial broadcast decoder forces each slot to reconstruct its own image region using frozen ViT features as targets. The JEPA prediction loss is disabled. Slots specialize into spatial regions.

**Stage 2 (40-100%):** Full training. JEPA latent prediction is activated alongside the SBD (now as a regularizer). Slots are already decomposed, so the dynamics model learns to predict object-specific futures.

This protocol resolves the tension between prediction efficiency (which favors distributed codes) and object specialization (which requires spatial grounding).

## Ablation Configs

Seven ablation variants test each architectural component:

```
configs/ablations/
  no_objects.yaml           # Flat JEPA baseline (1 slot)
  no_hierarchy.yaml         # Single-scale dynamics
  no_causality.yaml         # No causal graph
  no_ssm.yaml               # Transformer-only dynamics
  no_aux_decoder.yaml       # SIGReg only, no auxiliary decode
  no_memory.yaml            # No episodic memory
  no_event_detection.yaml   # All timesteps are events
```

## Known Limitations

These reflect the current state of the research, not fundamental architectural barriers.

**Slot count.** All 32 slots remain alive during training. The existence head does not learn to kill unused slots. Adaptive slot methods (AdaSlot, MetaSlot) are a path forward.

**Causal discovery.** The explicit causal adjacency matrix learns no edges under sparsity regularization. The GNN edge weights provide implicit interaction structure but have not been validated against ground-truth causal graphs.

**Scale.** Only the small config (68M) has been fully trained and evaluated. The base config (262M) has numerical stability issues at batch size >= 4 under bfloat16.

**Seed sensitivity.** Approximately 40-60% of training runs diverge to NaN within the first 1000 steps due to seed-dependent gradient overflow.

## Citation

```bibtex
@article{jaber2026hclsm,
  title={HCLSM: Hierarchical Causal Latent State Machines for Object-Centric World Modeling},
  author={Jaber, Jaber and Jaber, Osama},
  journal={arXiv preprint arXiv:2603.xxxxx},
  year={2026}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgments

Built at [RightNow AI](https://www.rightnowai.co/). Training compute provided by Nebius (NVIDIA H100 GPUs).
