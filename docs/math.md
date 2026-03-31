# HCLSM: Mathematical Foundations

## 1. World State Representation

The world state at time t is a structured latent state:

```
S_t = {
    objects: [o_1^t, o_2^t, ..., o_{N_t}^t],    N_t objects (variable count)
    relations: G_t = (V_t, E_t),                  Directed interaction graph
    context: c_t                                   Global scene context
}
```

Each object slot:
```
o_i^t ∈ R^{d_slot}     Latent state vector
p_i^t ∈ R^3            Predicted 3D position
v_i^t ∈ R^3            Predicted velocity
h_i^t ∈ R^{d_hidden}   Hidden state for SSM dynamics
alive_i^t ∈ [0, 1]     Existence probability (learned)
```

## 2. Slot Attention (Object Decomposition)

Given input tokens `X ∈ R^{B×M×d_input}` and N_max slot proposals, the iterative refinement:

```
slots^(0) ~ N(μ, σ²)                                    Learned Gaussian init
For iter = 1..n_iterations:
    q = W_q · LayerNorm(slots^(iter-1))                  (B, N, d_slot)
    k = W_k · LayerNorm(X)                               (B, M, d_slot)
    v = W_v · LayerNorm(X)                               (B, M, d_slot)

    A = softmax(q · k^T / √d_slot, dim=1)               OVER SLOT DIM (N)
    updates = A · v                                       (B, N, d_slot)
    slots^(iter) = GRU(updates, slots^(iter-1))
    slots^(iter) = slots^(iter) + MLP(LayerNorm(slots^(iter)))
```

**Key property:** `softmax(·, dim=1)` means each column of A sums to 1 — slots *compete* for tokens. This is the opposite of standard attention where queries compete for keys.

### Slot Birth/Death

```
alive_i = σ(W_exist · slots_i)                           Existence probability
residual_m = |1 - Σ_i A_{i,m}|                           Attention residual per token
If max_m(residual_m) > τ_birth AND ∃i: alive_i < τ_exist:
    slots_i ← W_birth · X_{argmax(residual)}             Birth from uncaptured token
```

## 3. Hierarchical Dynamics

### Level 0: Selective SSM (Continuous Physics)

Per-object Mamba-style selective state space model:

```
z_t = W_in · x_t                        Input projection
g_t = SiLU(W_gate · x_t)                Gating
Δ_t = softplus(W_Δ · z_t)               Input-dependent timestep
B_t = W_B · z_t                          Input-dependent state matrix
C_t = W_C · z_t                          Input-dependent output matrix

A = -exp(A_log)                          Learned log-scale state transition
dA_t = exp(Δ_t · A)                      Discretized state transition
dB_t = Δ_t · B_t                         Discretized input matrix

h_t = dA_t ⊙ h_{t-1} + dB_t ⊙ z_t      State update (linear recurrence)
y_t = (h_t · C_t).sum(-1)               Output projection
out_t = g_t ⊙ W_out · y_t + x_t         Gated output + residual
```

**Parallel scan:** The linear recurrence `h_t = gate_t · h_{t-1} + token_t` is associative:
```
(g_a, t_a) ⊕ (g_b, t_b) = (g_b · g_a, g_b · t_a + t_b)
```
Enabling O(log T) parallel depth via Blelloch scan.

### Level 1: Sparse Event Transformer

Event detection:
```
score_t = σ(MLP(Conv1d(states, window=w)))      Event probability per timestep
mask_t = 1[score_t > τ_event]                    Binary event mask
```

At event timesteps, all object slots attend to all objects:
```
For each event timestep k:
    states_k = MultiHeadAttn(states_k, states_k, states_k)    Standard Transformer
    states_k = states_k + SwiGLU(LayerNorm(states_k))
```

**Complexity:** O(K · N²) where K << T is the number of events.

### Level 2: Goal Dynamics (State Compression)

Cross-attention compression from K events to n_summary tokens:
```
summary = CrossAttn(query=learnable_tokens, key=events, value=events)
summary = Transformer(summary)
```

### Hierarchy Combination

```
final = Gate_0 · Level0 + Gate_1 · scatter(Level1) + Gate_2 · broadcast(Level2)
```
Where gates are learned per-level scalars.

## 4. Causal Graph

### Adjacency Matrix

```
A = σ(W_causal)                          N×N adjacency, A_ij ∈ [0,1]
```

### DAG Constraint (NOTEARS)

```
h(A) = tr(e^{A ⊙ A}) - N = 0            Acyclicity constraint
```

Where `e^M` is the matrix exponential. h(A) = 0 iff A is a DAG.

### Augmented Lagrangian

```
L_causal = λ_sparse · ||A||_1 + α · h(A) + (ρ/2) · h(A)²
```

With dual variable α updated via:
```
α ← α + ρ · h(A)
ρ ← min(ρ_max, γ · ρ)                    Penalty escalation
```

## 5. Loss Function

### Total Loss

```
L = λ_pred · L_prediction
  + λ_obj  · L_object
  + λ_aux  · L_auxiliary
  + λ_sig  · L_sigreg
  + λ_hier · L_hierarchy
  + λ_caus · L_causal
```

### L_prediction (Latent State Prediction, Primary)

```
L_prediction = (1/T) Σ_t ||f_pred(S_t) - sg(S_{t+1}^target)||²
```

Where `sg()` is stop-gradient, and the target encoder is EMA-updated:
```
θ_target ← τ · θ_target + (1-τ) · θ_online     τ = 0.996
```

### L_object (Slot Diversity + Tracking)

```
L_diversity = -(1/|pairs|) Σ_{i≠j} max(0, cos(z_i, z_j) - margin)
L_tracking  = (1/T) Σ_t Σ_i ||o_i^t - match(o_i^{t-1})||²
```

### L_auxiliary (Anti-Collapse Decoder)

```
L_auxiliary = ||D(pool(slots)) - crop(frames)||²
```

Where D is a 2-layer MLP and crop is a random 25% spatial crop. Intentionally weak — provides gradient pressure to retain information without encouraging full reconstruction.

### L_sigreg (Singular Value Regularization)

```
L_sigreg = Var(||z||) / (E[||z||²] + ε)
```

Simplified proxy for full SVD-based SIGReg. Penalizes low variance in embedding norms (collapse = all same norm).

### L_hierarchy (Cross-Level Consistency)

```
L_hierarchy = ||compress(mean_obj(Level0)) - mean(Level1)||²
```

Ensures coarse levels are consistent compressions of fine levels.

## 6. Default Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| λ_pred | 1.0 | Prediction loss weight |
| λ_obj | 0.5 | Object consistency weight |
| λ_aux | 0.1 | Auxiliary decode weight |
| λ_sigreg | 0.1 | SIGReg weight |
| λ_hierarchy | 0.25 | Hierarchy consistency weight |
| λ_causal | 0.1 | Causal regularization weight |
| τ (EMA) | 0.996 | Target encoder momentum |
| lr | 3e-4 | Peak learning rate |
| warmup | 10,000 | Linear warmup steps |
| total_steps | 500,000 | Total training steps |
| weight_decay | 0.05 | AdamW weight decay |
| max_grad_norm | 1.0 | Gradient clipping |
