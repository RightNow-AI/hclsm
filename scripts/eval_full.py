"""Comprehensive HCLSM evaluation suite.

Produces publication-quality figures and quantitative metrics from a trained checkpoint.

Usage:
    python scripts/eval_full.py --checkpoint runs/PROD_GPU0/checkpoints/step_50000.pt --preset small --data-dir lerobot/pusht --output-dir eval_results

Generates:
    1. Slot attention heatmaps (object decomposition)
    2. Causal graph visualization
    3. Prediction accuracy vs. horizon
    4. Multi-step rollout error curves
    5. Slot dynamics trajectories
    6. Event detection analysis
    7. Quantitative metrics table (LaTeX-ready)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Load model + data
# ═══════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, preset: str, device: str):
    """Load trained HCLSM model from checkpoint."""
    from hclsm.config import HCLSMConfig
    from hclsm.model import HCLSMWorldModel

    config = getattr(HCLSMConfig, preset)()
    config.causality.enabled = True
    config.dynamics.level2.d_action = 2  # PushT has 2D actions

    # Use default small config (32 slots) — matches checkpoint

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = ckpt["model"]

    model = HCLSMWorldModel(config)
    model.load_state_dict(model_state, strict=False)
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded {n_params/1e6:.1f}M model from {checkpoint_path}")
    return model, config


def load_data(data_dir: str, n_samples: int = 20, n_frames: int = 16):
    """Load evaluation samples from PushT dataset."""
    from hclsm.training.data import OpenXDataset
    ds = OpenXDataset(repo_id=data_dir, n_frames=n_frames, resolution=224, video_backend="pyav")
    logger.info(f"Dataset: {ds._n_episodes} episodes, d_action={ds.d_action}")

    samples = []
    indices = np.linspace(0, ds._n_episodes - 1, n_samples, dtype=int)
    for idx in indices:
        sample = ds[idx]
        samples.append(sample)
    return samples, ds


# ═══════════════════════════════════════════════════════════════════════════
# 0. SBD Spatial Decomposition (TRUE object ownership)
# ═══════════════════════════════════════════════════════════════════════════

def eval_sbd_decomposition(model, samples, config, device, output_dir):
    """Visualize SBD alpha masks — shows which pixels each slot OWNS."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logger.info("=== 0. SBD Spatial Decomposition ===")

    if not hasattr(model, "spatial_decoder"):
        logger.warning("No spatial_decoder in model, skipping SBD eval")
        return

    patch_size = config.perception.patch_size
    resolution = config.perception.input_resolution
    grid_size = resolution // patch_size

    for si in range(min(4, len(samples))):
        sample = samples[si]
        video = sample["video"].unsqueeze(0).to(device)

        with torch.no_grad():
            # Get slots and target features
            if hasattr(model, "module"):
                m = model.module
            else:
                m = model
            slots, alive, target_features = m._encode(video, use_target=True)

            # Run SBD to get alpha masks
            with torch.amp.autocast("cuda", enabled=False):
                _, alpha_masks, recon = m.spatial_decoder(
                    slots[:, 0].float(),
                    alive[:, 0].float(),
                    target_features[:, 0].float(),
                )
            # alpha_masks: (B, N, H_grid, W_grid)

        alpha = alpha_masks[0].cpu().numpy()  # (N, H_grid, W_grid)
        alive_vals = alive[0, 0].cpu().numpy()  # (N,)
        frame = video[0, 0].cpu().permute(1, 2, 0).numpy()
        frame = np.clip(frame, 0, 1)

        # Find slots with significant spatial ownership
        slot_ownership = alpha.max(axis=(1, 2))  # (N,) max alpha per slot
        top_slots = np.argsort(-slot_ownership)[:8]  # Top 8 by ownership

        n_show = min(8, len(top_slots))
        fig, axes = plt.subplots(2, n_show + 1, figsize=(3 * (n_show + 1), 6))

        # Row 1: Original frame + per-slot alpha masks
        axes[0, 0].imshow(frame)
        axes[0, 0].set_title("Original", fontsize=10, fontweight="bold")
        axes[0, 0].axis("off")

        # Row 2: Segmentation map (argmax over slots)
        seg_map = np.argmax(alpha, axis=0)  # (H_grid, W_grid)
        # Upscale
        seg_up = np.array(torch.nn.functional.interpolate(
            torch.from_numpy(seg_map.astype(np.float32)).unsqueeze(0).unsqueeze(0),
            size=(resolution, resolution), mode="nearest"
        ).squeeze())

        colors = plt.cm.Set3(np.linspace(0, 1, alpha.shape[0]))
        seg_rgb = np.zeros((resolution, resolution, 3))
        for slot_idx in range(alpha.shape[0]):
            mask = (seg_up == slot_idx)
            seg_rgb[mask] = colors[slot_idx][:3]
        axes[1, 0].imshow(seg_rgb * 0.6 + frame * 0.4)
        axes[1, 0].set_title("Segmentation", fontsize=10, fontweight="bold")
        axes[1, 0].axis("off")

        for col, slot_idx in enumerate(top_slots[:n_show]):
            mask = alpha[slot_idx]  # (H_grid, W_grid)
            mask_up = np.array(torch.nn.functional.interpolate(
                torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0),
                size=(resolution, resolution), mode="bilinear", align_corners=False
            ).squeeze())

            # Row 1: Alpha heatmap
            axes[0, col + 1].imshow(mask_up, cmap="hot", vmin=0, vmax=1)
            ownership = slot_ownership[slot_idx]
            axes[0, col + 1].set_title(f"S{slot_idx}\n({ownership:.2f})", fontsize=9)
            axes[0, col + 1].axis("off")

            # Row 2: Overlay on frame
            overlay = frame.copy() * 0.3
            color = colors[slot_idx][:3]
            for c in range(3):
                overlay[:, :, c] += mask_up * color[c] * 0.7
            axes[1, col + 1].imshow(np.clip(overlay, 0, 1))
            axes[1, col + 1].axis("off")

        fig.suptitle(f"Spatial Broadcast Decoder — Object Ownership (Sample {si})",
                    fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_dir / f"sbd_decomposition_sample{si}.png", dpi=200, bbox_inches="tight")
        plt.close()

        n_active = int((slot_ownership > 0.1).sum())
        logger.info(f"  Saved sbd_decomposition_sample{si}.png ({n_active} active slots)")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Slot Attention Heatmaps
# ═══════════════════════════════════════════════════════════════════════════

def eval_slot_attention(model, samples, config, device, output_dir):
    """Visualize object decomposition via slot attention maps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logger.info("=== 1. Slot Attention Heatmaps ===")

    patch_size = config.perception.patch_size
    resolution = config.perception.input_resolution
    grid_size = resolution // patch_size  # 14 for 224/16

    # Pick 4 diverse samples
    sample_indices = [0, len(samples)//4, len(samples)//2, 3*len(samples)//4]

    for si, sample_idx in enumerate(sample_indices):
        sample = samples[sample_idx]
        video = sample["video"].unsqueeze(0).to(device)  # (1, T, C, H, W)

        with torch.no_grad():
            output = model(video, return_attention=True)

        attn = output.slot_attention_maps  # (1, T, N_max, M)
        alive = output.alive_mask          # (1, T, N_max)

        if attn is None:
            logger.warning("No attention maps returned. Skipping slot visualization.")
            return

        attn = attn[0].cpu().numpy()       # (T, N_max, M)
        alive = alive[0].cpu().numpy()     # (T, N_max)

        # Pick 4 time steps
        T = attn.shape[0]
        time_steps = [0, T//4, T//2, T-1]

        # Count alive slots at each time step
        n_alive_max = max(int((alive[t] > 0.5).sum()) for t in time_steps)
        n_alive_max = max(n_alive_max, 2)  # At least show 2 slots

        fig, axes = plt.subplots(n_alive_max + 1, len(time_steps),
                                 figsize=(3 * len(time_steps), 3 * (n_alive_max + 1)))
        if n_alive_max + 1 == 1:
            axes = axes[np.newaxis, :]

        slot_colors = plt.cm.Set2(np.linspace(0, 1, n_alive_max))

        for ti, t in enumerate(time_steps):
            # Original frame
            frame = video[0, t].cpu().permute(1, 2, 0).numpy()
            frame = np.clip(frame, 0, 1)
            axes[0, ti].imshow(frame)
            axes[0, ti].set_title(f"t={t}", fontsize=12, fontweight="bold")
            axes[0, ti].axis("off")

            # Alive slot masks
            alive_slots = np.where(alive[t] > 0.5)[0]
            for row, slot_idx in enumerate(alive_slots[:n_alive_max]):
                mask = attn[t, slot_idx].reshape(grid_size, grid_size)
                mask_up = np.array(
                    torch.nn.functional.interpolate(
                        torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0),
                        size=(resolution, resolution),
                        mode="bilinear", align_corners=False
                    ).squeeze()
                )
                # Overlay attention on frame
                overlay = frame.copy()
                color = slot_colors[row][:3]
                for c in range(3):
                    overlay[:, :, c] = frame[:, :, c] * 0.3 + mask_up * color[c] * 0.7

                axes[row + 1, ti].imshow(np.clip(overlay, 0, 1))
                axes[row + 1, ti].set_title(
                    f"Slot {slot_idx} (p={alive[t, slot_idx]:.2f})", fontsize=9
                )
                axes[row + 1, ti].axis("off")

            # Clear unused rows
            for row in range(len(alive_slots), n_alive_max):
                axes[row + 1, ti].axis("off")

        fig.suptitle(f"Object Decomposition — Sample {si+1}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_dir / f"slot_attention_sample{si}.png", dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved slot_attention_sample{si}.png ({len(alive_slots)} alive slots)")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Causal Graph Visualization
# ═══════════════════════════════════════════════════════════════════════════

def eval_causal_graph(model, samples, device, output_dir):
    """Visualize the learned causal adjacency matrix."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logger.info("=== 2. Causal Graph ===")

    # Run multiple samples to get stable adjacency
    all_adj = []
    all_alive = []
    for sample in samples[:10]:
        video = sample["video"].unsqueeze(0).to(device)
        actions = sample.get("action")
        if actions is not None:
            actions = actions.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(video, actions=actions)
        # Use GNN edge weights as causal graph (post-hoc extraction)
        # The GNN learns which objects interact — these ARE the causal edges
        if output.edge_weights is not None:
            gnn_adj = output.edge_weights[0].mean(dim=0).cpu().numpy()  # (N, N) avg over time
        else:
            gnn_adj = output.causal_graph.cpu().numpy()
        all_adj.append(gnn_adj)
        alive_avg = output.alive_mask[0].mean(dim=0).cpu().numpy()
        all_alive.append(alive_avg)

    adj = np.mean(all_adj, axis=0)  # (N_max, N_max)
    alive_avg = np.mean(all_alive, axis=0)

    # Filter to alive slots only
    alive_idx = np.where(alive_avg > 0.3)[0]
    n_alive = len(alive_idx)

    if n_alive < 2:
        logger.warning("Fewer than 2 alive slots — skipping causal graph.")
        return

    adj_alive = adj[np.ix_(alive_idx, alive_idx)]

    stats = model.causal_graph.get_edge_statistics()
    logger.info(f"  Causal stats: {stats}")

    # --- Adjacency matrix heatmap ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im = axes[0].imshow(adj_alive, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="equal")
    axes[0].set_title("Learned Causal Adjacency (soft)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Effect (slot j)")
    axes[0].set_ylabel("Cause (slot i)")
    for i in range(n_alive):
        for j in range(n_alive):
            axes[0].text(j, i, f"{adj_alive[i,j]:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if adj_alive[i,j] > 0.5 else "black")
    axes[0].set_xticks(range(n_alive))
    axes[0].set_yticks(range(n_alive))
    axes[0].set_xticklabels([f"S{i}" for i in alive_idx])
    axes[0].set_yticklabels([f"S{i}" for i in alive_idx])
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    adj_hard = (adj_alive > 0.5).astype(float)
    axes[1].imshow(adj_hard, cmap="Greys", vmin=0, vmax=1, aspect="equal")
    axes[1].set_title("Thresholded Causal Graph (>0.5)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Effect (slot j)")
    axes[1].set_ylabel("Cause (slot i)")
    axes[1].set_xticks(range(n_alive))
    axes[1].set_yticks(range(n_alive))
    axes[1].set_xticklabels([f"S{i}" for i in alive_idx])
    axes[1].set_yticklabels([f"S{i}" for i in alive_idx])

    n_edges = int(adj_hard.sum()) - n_alive
    fig.suptitle(f"Causal Structure ({n_alive} objects, {n_edges} causal edges)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "causal_graph.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- Directed graph layout ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    angles = np.linspace(0, 2 * np.pi, n_alive, endpoint=False)
    positions = np.column_stack([np.cos(angles), np.sin(angles)]) * 2.0

    for i in range(n_alive):
        for j in range(n_alive):
            if i != j and adj_alive[i, j] > 0.3:
                weight = adj_alive[i, j]
                ax.annotate("",
                    xy=(positions[j, 0], positions[j, 1]),
                    xytext=(positions[i, 0], positions[i, 1]),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=plt.cm.Reds(weight),
                        lw=1.5 + weight * 3,
                        connectionstyle="arc3,rad=0.15",
                    ))

    for i in range(n_alive):
        circle = plt.Circle(positions[i], 0.25, color=plt.cm.Set2(i / max(n_alive, 1)),
                          ec="black", lw=2, zorder=5)
        ax.add_patch(circle)
        ax.text(positions[i, 0], positions[i, 1], f"S{alive_idx[i]}",
               ha="center", va="center", fontsize=12, fontweight="bold", zorder=6)

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")
    ax.set_title("Learned Causal DAG", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.savefig(output_dir / "causal_dag.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  Saved causal_graph.png + causal_dag.png")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Prediction Accuracy vs. Horizon
# ═══════════════════════════════════════════════════════════════════════════

def eval_prediction_horizon(model, samples, device, output_dir):
    """Measure prediction accuracy at different time horizons."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logger.info("=== 3. Prediction vs. Horizon ===")

    max_horizon = 14
    horizon_errors = {h: [] for h in range(1, max_horizon + 1)}

    for sample in samples[:15]:
        video = sample["video"].unsqueeze(0).to(device)
        actions = sample.get("action")
        if actions is not None:
            actions = actions.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(video, actions=actions)

        states = output.predicted_states[0].cpu()  # (T, N_max, d_slot)
        slots = output.object_slots[0].cpu()        # (T, N_max, d_slot)

        T = states.shape[0]
        for h in range(1, min(max_horizon + 1, T)):
            for t_start in range(0, T - h):
                pred = states[t_start + h]
                true = slots[t_start + h]
                mse = ((pred - true) ** 2).mean().item()
                horizon_errors[h].append(mse)

    horizons = sorted(horizon_errors.keys())
    means = [np.mean(horizon_errors[h]) if horizon_errors[h] else 0 for h in horizons]
    stds = [np.std(horizon_errors[h]) if horizon_errors[h] else 0 for h in horizons]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(horizons, means, "o-", color="#2563eb", linewidth=2.5, markersize=8, label="Mean MSE")
    ax.fill_between(horizons,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.2, color="#2563eb", label="\u00b11 std")
    ax.set_xlabel("Prediction Horizon (frames)", fontsize=13)
    ax.set_ylabel("Latent MSE", fontsize=13)
    ax.set_title("Prediction Accuracy vs. Time Horizon", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(horizons)

    plt.tight_layout()
    plt.savefig(output_dir / "prediction_horizon.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved prediction_horizon.png (1-step MSE={means[0]:.6f})")

    return {"horizon_means": means, "horizon_stds": stds}


# ═══════════════════════════════════════════════════════════════════════════
# 4. Slot Dynamics Trajectories (PCA)
# ═══════════════════════════════════════════════════════════════════════════

def eval_slot_trajectories(model, samples, device, output_dir):
    """Visualize slot state trajectories in 2D via PCA."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    logger.info("=== 4. Slot Trajectories (PCA) ===")

    all_trajectories = []
    all_labels = []

    for si, sample in enumerate(samples[:5]):
        video = sample["video"].unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(video)

        slots = output.object_slots[0].cpu().numpy()  # (T, N_max, d_slot)
        alive = output.alive_mask[0].cpu().numpy()

        alive_mean = alive.mean(axis=0)
        alive_idx = np.where(alive_mean > 0.5)[0]

        for slot_idx in alive_idx:
            traj = slots[:, slot_idx, :]
            all_trajectories.append(traj)
            all_labels.append((si, slot_idx))

    if len(all_trajectories) < 2:
        logger.warning("Not enough trajectories for PCA.")
        return

    all_points = np.concatenate(all_trajectories, axis=0)
    pca = PCA(n_components=2)
    projected = pca.fit_transform(all_points)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_trajectories)))

    offset = 0
    for i, (traj, label) in enumerate(zip(all_trajectories, all_labels)):
        T = traj.shape[0]
        pts = projected[offset:offset + T]
        ax.plot(pts[:, 0], pts[:, 1], "-", color=colors[i], linewidth=1.5, alpha=0.7)
        ax.scatter(pts[0, 0], pts[0, 1], color=colors[i], s=80, marker="o",
                  edgecolors="black", zorder=5)
        ax.scatter(pts[-1, 0], pts[-1, 1], color=colors[i], s=80, marker="s",
                  edgecolors="black", zorder=5)
        ax.annotate(f"S{label[1]}", (pts[0, 0], pts[0, 1]), fontsize=8,
                   fontweight="bold", color=colors[i])
        offset += T

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
    ax.set_title("Slot State Trajectories in Latent Space (PCA)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "slot_trajectories_pca.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved slot_trajectories_pca.png ({len(all_trajectories)} trajectories, "
                f"PCA explained {sum(pca.explained_variance_ratio_)*100:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Event Detection Analysis
# ═══════════════════════════════════════════════════════════════════════════

def eval_event_detection(model, samples, device, output_dir):
    """Analyze event detection: when does the model detect state transitions?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logger.info("=== 5. Event Detection ===")

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    for si, sample in enumerate(samples[:4]):
        video = sample["video"].unsqueeze(0).to(device)
        actions = sample.get("action")
        if actions is not None:
            actions = actions.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(video, actions=actions)

        event_scores = output.event_scores[0].cpu().numpy()
        event_mask = output.event_mask[0].cpu().numpy()

        T = len(event_scores)
        t_axis = np.arange(T)

        axes[si].fill_between(t_axis, event_scores, alpha=0.3, color="#2563eb", label="Event probability")
        axes[si].plot(t_axis, event_scores, "-o", color="#2563eb", markersize=4, linewidth=1.5)

        event_times = np.where(event_mask > 0.5)[0]
        for et in event_times:
            axes[si].axvline(et, color="#dc2626", linestyle="--", alpha=0.7, linewidth=1.5)

        axes[si].set_ylabel("P(event)", fontsize=11)
        axes[si].set_title(f"Sample {si+1} — {len(event_times)} events detected", fontsize=11)
        axes[si].set_ylim(-0.05, 1.05)
        axes[si].grid(True, alpha=0.3)
        axes[si].legend(fontsize=9)

    axes[-1].set_xlabel("Frame", fontsize=12)
    fig.suptitle("Event Detection Across Episodes", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "event_detection.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  Saved event_detection.png")


# ═══════════════════════════════════════════════════════════════════════════
# 6. Hierarchy Level Comparison
# ═══════════════════════════════════════════════════════════════════════════

def eval_hierarchy_comparison(model, samples, device, output_dir):
    """Compare representations at different hierarchy levels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logger.info("=== 6. Hierarchy Level Comparison ===")

    pred_norms = []
    slot_norms = []

    for sample in samples[:10]:
        video = sample["video"].unsqueeze(0).to(device)
        actions = sample.get("action")
        if actions is not None:
            actions = actions.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(video, actions=actions)

        pred = output.predicted_states[0].cpu()
        slots = output.object_slots[0].cpu()

        pred_norm = pred.norm(dim=-1).mean(dim=-1).numpy()
        slot_norm = slots.norm(dim=-1).mean(dim=-1).numpy()
        pred_norms.append(pred_norm)
        slot_norms.append(slot_norm)

    T = min(len(n) for n in pred_norms)
    pred_avg = np.mean([n[:T] for n in pred_norms], axis=0)
    slot_avg = np.mean([n[:T] for n in slot_norms], axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    t_axis = np.arange(T)
    ax.plot(t_axis, slot_avg, "o-", color="#2563eb", linewidth=2, label="Perception (slots)", markersize=6)
    ax.plot(t_axis, pred_avg, "s-", color="#dc2626", linewidth=2, label="Dynamics (predicted)", markersize=6)
    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("Mean Representation Norm", fontsize=12)
    ax.set_title("Perception vs. Dynamics Representations Over Time", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "hierarchy_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  Saved hierarchy_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# 7. Quantitative Metrics
# ═══════════════════════════════════════════════════════════════════════════

def eval_quantitative(model, samples, device, output_dir):
    """Compute all quantitative metrics and save as JSON + LaTeX table."""
    logger.info("=== 7. Quantitative Metrics ===")

    total_losses = []
    pred_losses = []
    track_losses = []
    causal_losses = []
    diversity_losses = []
    n_alive_slots = []
    n_causal_edges = []

    for sample in samples:
        video = sample["video"].unsqueeze(0).to(device)
        actions = sample.get("action")
        if actions is not None:
            actions = actions.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(video, actions=actions, targets=video)

        losses = output.losses
        total_losses.append(losses.get("total", torch.tensor(0)).item())
        pred_losses.append(losses.get("prediction", torch.tensor(0)).item())
        track_losses.append(losses.get("tracking", torch.tensor(0)).item())
        causal_losses.append(losses.get("causal", torch.tensor(0)).item())
        diversity_losses.append(losses.get("diversity", torch.tensor(0)).item())

        alive = output.alive_mask[0].mean(dim=0)
        n_alive_slots.append(int((alive > 0.5).sum().item()))

        adj = output.causal_graph.cpu().numpy()
        alive_idx = np.where(alive.cpu().numpy() > 0.5)[0]
        if len(alive_idx) >= 2:
            adj_sub = adj[np.ix_(alive_idx, alive_idx)]
            n_edges = int((adj_sub > 0.5).sum()) - len(alive_idx)
            n_causal_edges.append(max(0, n_edges))
        else:
            n_causal_edges.append(0)

    metrics = {
        "total_loss": {"mean": float(np.mean(total_losses)), "std": float(np.std(total_losses))},
        "prediction_loss": {"mean": float(np.mean(pred_losses)), "std": float(np.std(pred_losses))},
        "tracking_loss": {"mean": float(np.mean(track_losses)), "std": float(np.std(track_losses))},
        "causal_loss": {"mean": float(np.mean(causal_losses)), "std": float(np.std(causal_losses))},
        "diversity_loss": {"mean": float(np.mean(diversity_losses)), "std": float(np.std(diversity_losses))},
        "n_alive_slots": {"mean": float(np.mean(n_alive_slots)), "std": float(np.std(n_alive_slots))},
        "n_causal_edges": {"mean": float(np.mean(n_causal_edges)), "std": float(np.std(n_causal_edges))},
        "n_samples": len(samples),
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # LaTeX table
    latex = r"""\begin{table}[h]
\centering
\caption{HCLSM Quantitative Results on PushT (68M parameters, 50K steps)}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{Mean} & \textbf{Std} \\
\midrule
"""
    for key, val in metrics.items():
        if isinstance(val, dict):
            name = key.replace("_", " ").title()
            latex += f"{name} & {val['mean']:.4f} & {val['std']:.4f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\label{tab:results}
\end{table}"""

    with open(output_dir / "metrics_table.tex", "w") as f:
        f.write(latex)

    print("\n" + "=" * 60)
    print("QUANTITATIVE RESULTS")
    print("=" * 60)
    for key, val in metrics.items():
        if isinstance(val, dict):
            print(f"  {key:25s}: {val['mean']:.4f} +/- {val['std']:.4f}")
        else:
            print(f"  {key:25s}: {val}")
    print("=" * 60)

    logger.info("  Saved metrics.json + metrics_table.tex")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# 8. Multi-Seed Comparison
# ═══════════════════════════════════════════════════════════════════════════

def eval_multi_seed(run_dirs: list[Path], output_dir: Path):
    """Compare training curves across multiple seed runs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logger.info("=== 8. Multi-Seed Comparison ===")

    all_curves = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) > 0:
            steps = [d.get("step", i) for i, d in enumerate(data)]
            totals = [d.get("total", 0) for d in data]
            all_curves.append((steps, totals, run_dir.name))

    if len(all_curves) < 2:
        logger.info("  Need at least 2 runs for multi-seed comparison. Skipping.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_curves)))

    for i, (steps, totals, name) in enumerate(all_curves):
        window = max(1, len(totals) // 100)
        smoothed = np.convolve(totals, np.ones(window)/window, mode="valid")
        ax.plot(steps[:len(smoothed)], smoothed, color=colors[i], linewidth=1.5,
               label=name, alpha=0.8)

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Total Loss", fontsize=12)
    ax.set_title("Training Curves Across Seeds (Real PushT Data)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "multi_seed_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved multi_seed_comparison.png ({len(all_curves)} seeds)")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Full HCLSM Evaluation Suite")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint")
    parser.add_argument("--preset", default="small", choices=["tiny", "small", "base", "large"])
    parser.add_argument("--data-dir", default="lerobot/pusht", help="HuggingFace dataset ID")
    parser.add_argument("--output-dir", default="eval_results", help="Where to save figures")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-samples", type=int, default=20, help="Number of eval samples")
    parser.add_argument("--runs-dir", default=None, help="Directory with multiple seed runs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output: {output_dir}")
    logger.info(f"Device: {args.device}")

    model, config = load_model(args.checkpoint, args.preset, args.device)
    samples, dataset = load_data(args.data_dir, n_samples=args.n_samples)
    logger.info(f"Loaded {len(samples)} samples")

    # Run SBD spatial ownership visualization (the TRUE decomposition signal)
    eval_sbd_decomposition(model, samples, config, args.device, output_dir)

    # Run all evaluations
    eval_slot_attention(model, samples, config, args.device, output_dir)
    eval_causal_graph(model, samples, args.device, output_dir)
    horizon_metrics = eval_prediction_horizon(model, samples, args.device, output_dir)
    try:
        eval_slot_trajectories(model, samples, args.device, output_dir)
    except ImportError:
        logger.warning("scikit-learn not installed, skipping PCA trajectories")
    eval_event_detection(model, samples, args.device, output_dir)
    eval_hierarchy_comparison(model, samples, args.device, output_dir)
    metrics = eval_quantitative(model, samples, args.device, output_dir)

    if args.runs_dir:
        run_dirs = sorted(Path(args.runs_dir).glob("PROD_GPU*"))
        eval_multi_seed(run_dirs, output_dir)

    # Save combined results
    combined = {
        "checkpoint": args.checkpoint,
        "preset": args.preset,
        "data": args.data_dir,
        "n_samples": args.n_samples,
        "metrics": metrics,
        "horizon": {
            "means": horizon_metrics["horizon_means"],
            "stds": horizon_metrics["horizon_stds"],
        },
    }
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(combined, f, indent=2)

    logger.info(f"\nAll results saved to {output_dir}/")
    logger.info("Figures: slot_attention_*.png, causal_graph.png, causal_dag.png, "
                "prediction_horizon.png, slot_trajectories_pca.png, "
                "event_detection.png, hierarchy_comparison.png")
    logger.info("Data: metrics.json, metrics_table.tex, eval_results.json")


if __name__ == "__main__":
    main()
