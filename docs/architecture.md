# HCLSM: Architecture & Implementation

## Overview

The **Hierarchical Causal Latent State Machine (HCLSM)** is a world model architecture that operates on five interconnected layers: Perception, Object Decomposition, Hierarchical Dynamics, Causal Reasoning, and Continual Memory. Unlike flat latent-prediction models (V-JEPA, VL-JEPA), HCLSM reasons about the world through structured object slots, three-level temporal dynamics, and an explicit causal graph.

```
Input Video (B, T, C, H, W)
        │
        ▼
┌──────────────────────┐
│ Layer 1: Perception   │  ViT encoder + multimodal fuser
│ (VisionEncoder)       │  (B,T,C,H,W) → (B,T,M,d_world)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Layer 2: Objects      │  Dynamic Slot Attention + GNN
│ (DynamicSlotAttention │  (B,T,M,d_world) → (B,T,N,d_slot)
│  + RelationGraph)     │  Variable-count slots with birth/death
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Layer 3: Dynamics     │  Three-level hierarchy:
│  Level 0: SSM         │    SSM (continuous physics, per-object)
│  Level 1: Events      │    Sparse Transformer (discrete events)
│  Level 2: Goals       │    Compressed Transformer (abstract plans)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Layer 4: Causality    │  Explicit causal adjacency matrix
│ (CausalGraph)         │  DAG-constrained, Gumbel-softmax edges
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Layer 5: Memory       │  Episodic (Hopfield) + Semantic
│ (Consolidation)       │  Sleep-phase distillation
└──────────────────────┘
```

## Model Configurations

| Variant | Parameters | d_world | d_slot | N_max | SSM blocks | L1 layers | L2 layers |
|---------|-----------|---------|--------|-------|------------|-----------|-----------|
| Tiny    | ~50M      | 256     | 128    | 16    | 2          | 2         | 2         |
| Small   | ~200M     | 384     | 192    | 32    | 3          | 3         | 3         |
| Base    | ~800M     | 768     | 256    | 64    | 4          | 4         | 6         |
| Large   | ~3B       | 1024    | 384    | 128   | 6          | 6         | 8         |

## Key Architectural Decisions

1. **Predict in latent space, not data space.** HCLSM is a world model, not a video generator. Full frame reconstruction is only used as a weak auxiliary loss for anti-collapse.

2. **Objects are first-class citizens.** Every component operates on object slots, not flat feature maps. Slot attention decomposes scenes with learned birth/death of variable-count objects.

3. **Hierarchy is structural, not learned.** The three levels (SSM / event Transformer / goal Transformer) are architecturally distinct. Level 0 handles continuous physics, Level 1 fires at discrete event boundaries, Level 2 compresses across events for abstract planning.

4. **SSM for continuous, Transformer for discrete.** Mamba-style SSM runs per-object at every timestep (cheap, O(T) sequential). Sparse Transformer fires only at detected events (expensive per event, but K << T events).

5. **Causal graph is explicit.** An actual N_max x N_max adjacency matrix, not implicit in attention patterns. Must be inspectable, differentiable, and DAG-constrained.

6. **Anti-collapse is dual.** SIGReg for statistical health (prevents mode collapse in embedding space) + auxiliary decoder for information preservation (prevents semantic collapse). Together they are more robust than either alone.

---

## Layer 1: Perception

**Module:** `hclsm/perception/vision_encoder.py`

ViT-based encoder with optional initialization from V-JEPA 2 checkpoints. Processes video frames through patch embedding + temporal position encoding + transformer blocks.

- Input: `(B, T, C, H, W)` video frames
- Output: `(B, T, M, d_model)` patch embeddings where M = (H/patch_size)^2
- Supports gradient checkpointing for memory efficiency
- MultimodalFuser projects from d_model to d_world for downstream layers

## Layer 2: Object Decomposition

**Module:** `hclsm/objects/dynamic_slots.py`

The core innovation over standard Slot Attention: **variable-count objects** with learned slot birth/death.

### Dynamic Slot Attention

```
Initialize N_max slot proposals from learned Gaussian(mu, sigma)
for iteration in range(n_iterations):
    q = project_q(LayerNorm(slots))          # (B, N, d_slot)
    logits = q @ k^T * scale                 # (B, N, M)
    attn = softmax(logits, dim=1)            # CRITICAL: over SLOT dim
    updates = attn @ v                       # (B, N, d_slot)
    slots = GRU(updates, slots)              # (B, N, d_slot)
    slots = slots + MLP(LayerNorm(slots))    # residual refinement

alive = sigmoid(existence_head(slots))       # (B, N)
```

**Critical detail:** The softmax is over dim=1 (slot dimension), creating *competition between slots* for input tokens. This is the opposite of standard attention where softmax is over the key dimension.

### Slot Birth/Death

- **Death:** Slots with `alive < existence_threshold` are considered dead
- **Birth:** When residual attention energy (tokens not captured by any slot) exceeds `birth_threshold`, a dormant slot is initialized from the highest-residual token via a learned projection

### Relation Graph (GNN)

**Module:** `hclsm/objects/relation_graph.py`

Models pairwise object interactions:
- Edge features: `e_ij = MLP([o_i; o_j; o_i - o_j; o_i * o_j])`
- Edge weights: `w_ij = sigmoid(linear(e_ij))`
- Message passing: `m_ij = w_ij * MLP(e_ij)`, aggregation via sum
- Node update: `o_i' = o_i + MLP([o_i; sum_j(m_ij)])`
- Multiple rounds of message passing (default: 2)

## Layer 3: Hierarchical Dynamics

### Level 0: SSM (Continuous Physics)

**Module:** `hclsm/dynamics/level0_ssm.py`

Per-object Mamba-style selective SSM for smooth trajectories. Each object gets its own SSM track (shared parameters). A separate global SSM processes mean-pooled object states and conditions the per-object tracks.

```
Global SSM: mean_pool(objects) → global_blocks → condition_proj
Per-object: objects + global_condition → object_blocks → predicted
```

Uses mamba-ssm when available, falls back to SimplifiedSSMBlock (pure PyTorch sequential scan) otherwise.

### Level 1: Event Dynamics (Sparse Transformer)

**Module:** `hclsm/dynamics/level1_event.py`

Fires **only at detected event timesteps**. Within each event, all object slots attend to all objects via standard multi-head attention with SwiGLU FFN.

- Event detection via `EventDetector`: Conv1d over temporal window + MLP
- Sparse: processes K << T timesteps, so cost is O(K * N^2) not O(T * N^2)

### Level 2: Goal Dynamics (Compressed Transformer)

**Module:** `hclsm/dynamics/level2_goal.py`

Compresses event-level representations into a small set of summary tokens via cross-attention, then processes with a transformer for abstract goal-level reasoning.

### Hierarchy Manager

**Module:** `hclsm/dynamics/hierarchy_manager.py`

Gathers event states from the temporal grid, manages cross-level communication, and combines predictions from all three levels via learned gating.

## Layer 4: Causality

**Module:** `hclsm/causality/causal_graph.py`

Learns an explicit N_max x N_max causal adjacency matrix over object slots. Parameterized as `A = sigmoid(W_causal)` with:
- L1 sparsity regularization
- NOTEARS DAG constraint: `tr(e^{A ⊙ A}) - d = 0`
- Augmented Lagrangian optimization

## Training

### JEPA-Style Self-Supervision

HCLSM uses self-supervised learning with an EMA target encoder:
1. Online encoder processes input video → slot predictions
2. Target encoder (EMA-updated, stop-gradient) produces targets
3. Loss: predict target slots at t+1 from online slots at t

### Multi-Objective Loss

```
L_total = λ_pred * L_prediction       (latent next-state prediction)
        + λ_obj  * L_object            (slot diversity + temporal tracking)
        + λ_aux  * L_auxiliary          (random-crop decode anti-collapse)
        + λ_sigreg * L_sigreg          (SIGReg embedding health)
        + λ_hierarchy * L_hierarchy    (cross-level consistency)
        + λ_causal * L_causal          (DAG sparsity + acyclicity)
```

### Gradient Accumulation & Distributed Training

- Configurable gradient accumulation steps
- FSDP wrapping with configurable sharding strategy (FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD)
- FSDP-aware checkpoint save/load
- DistributedSampler with epoch-aware shuffling
- Mixed precision (bf16 compute, fp32 reduce/buffer)
