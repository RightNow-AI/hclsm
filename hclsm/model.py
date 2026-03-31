"""Top-level HCLSM World Model.

Wires all five layers together: Perception -> Objects -> Dynamics ->
Causality -> Memory. Manages EMA target encoder for JEPA-style training.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from hclsm.config import HCLSMConfig
from hclsm.perception.vision_encoder import VisionEncoder
from hclsm.perception.multimodal_fuser import MultimodalFuser
from hclsm.objects.dynamic_slots import DynamicSlotAttention
from hclsm.objects.slot_tracker import SlotTracker
from hclsm.objects.relation_graph import RelationGraph
from hclsm.objects.losses import slot_diversity_loss, slot_tracking_loss
from hclsm.objects.spatial_decoder import SpatialBroadcastDecoder
from hclsm.dynamics.level0_ssm import Level0SSM
from hclsm.dynamics.event_detector import EventDetector
from hclsm.dynamics.level1_event import Level1EventDynamics
from hclsm.dynamics.level2_goal import Level2GoalDynamics
from hclsm.dynamics.hierarchy_manager import HierarchyManager
from hclsm.causality.causal_graph import CausalGraph
from hclsm.causality.action_conditioner import ActionConditioner
from hclsm.training.losses import CombinedLoss


@dataclass
class WorldModelOutput:
    """Output of HCLSMWorldModel.forward()."""

    predicted_states: torch.Tensor  # (B, T, N_max, d_slot)
    losses: dict[str, torch.Tensor] = field(default_factory=dict)
    object_slots: torch.Tensor | None = None  # (B, T, N_max, d_slot)
    alive_mask: torch.Tensor | None = None  # (B, T, N_max)
    event_mask: torch.Tensor | None = None  # (B, T)
    event_scores: torch.Tensor | None = None  # (B, T)
    causal_graph: torch.Tensor | None = None  # (N_max, N_max)
    abstract_states: torch.Tensor | None = None  # (B, n_summary, d_l2)
    edge_weights: torch.Tensor | None = None  # (B, T, N_max, N_max)
    slot_attention_maps: torch.Tensor | None = None  # (B, T, N_max, M)


class HCLSMWorldModel(nn.Module):
    """Hierarchical Causal Latent State Machine — the full architecture.

    Five layers:
    1. Perception: encode video frames to patch embeddings
    2. Objects: decompose into variable-count object slots via slot attention
    3. Dynamics: three-level hierarchical temporal prediction (SSM + Transformers)
    4. Causality: learn causal graph over objects
    5. Memory: episodic memory for continual learning

    Training uses JEPA-style self-supervision with an EMA target encoder.
    """

    def __init__(self, config: HCLSMConfig) -> None:
        super().__init__()
        self.config = config

        # ─── Layer 1: Perception ───
        self.perception = VisionEncoder(config.perception)
        self.fuser = MultimodalFuser(config)

        # ─── Layer 2: Objects ───
        self.slot_attention = DynamicSlotAttention(config.objects, config.d_world)
        self.slot_tracker = SlotTracker(config.objects)
        self.relation_graph = RelationGraph(config.objects)

        # Object property heads
        d_slot = config.objects.d_slot
        self.position_head = nn.Linear(d_slot, 3)
        self.velocity_head = nn.Linear(d_slot, 3)

        # ─── Layer 3: Dynamics ───
        self.level0_ssm = Level0SSM(config.dynamics.level0, d_slot)
        self.event_detector = EventDetector(
            d_input=d_slot,
            window_size=config.dynamics.event_window_size,
            threshold=config.dynamics.event_threshold,
        )
        self.level1_event = Level1EventDynamics(config.dynamics.level1, d_slot)
        self.level2_goal = Level2GoalDynamics(config.dynamics.level2, d_slot)
        self.hierarchy_manager = HierarchyManager(
            d_slot, config.dynamics.level2.d_model,
        )

        # ─── Spatial Broadcast Decoder (forces slot → object decomposition) ───
        grid_size = config.perception.input_resolution // config.perception.patch_size
        self.spatial_decoder = SpatialBroadcastDecoder(
            d_slot=d_slot,
            d_target=config.d_world,
            grid_size=grid_size,
            hidden_dim=min(d_slot * 2, 256),
            n_layers=4,
        )

        # ─── Action conditioning (for robotics data) ───
        self.d_action = config.dynamics.level2.d_action
        if self.d_action > 0:
            self.action_conditioner = ActionConditioner(
                self.d_action, d_slot, config.objects.n_max_slots,
            )

        # ─── Layer 4: Causality ───
        self.causal_graph = CausalGraph(
            config.causality, config.objects.n_max_slots,
        )

        # ─── Loss computation ───
        self.loss_fn = CombinedLoss(config.training, d_slot)

        # ─── EMA Target Encoder ───
        self.target_perception = copy.deepcopy(self.perception)
        self.target_fuser = copy.deepcopy(self.fuser)
        self.target_slot_attention = copy.deepcopy(self.slot_attention)
        for param in self.target_parameters():
            param.requires_grad = False

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing on all supported modules."""
        self.perception.use_gradient_checkpointing = True
        self.slot_attention.use_gradient_checkpointing = True
        self.level0_ssm.use_gradient_checkpointing = True
        self.level1_event.use_gradient_checkpointing = True

    def target_parameters(self):
        """Iterator over all target encoder parameters."""
        yield from self.target_perception.parameters()
        yield from self.target_fuser.parameters()
        yield from self.target_slot_attention.parameters()

    def online_parameters(self):
        """Iterator over online encoder parameters (matched to target)."""
        yield from self.perception.parameters()
        yield from self.fuser.parameters()
        yield from self.slot_attention.parameters()

    @torch.no_grad()
    def update_target_encoder(self, decay: float = 0.996) -> None:
        """EMA update of target encoder from online encoder."""
        for online_p, target_p in zip(
            self.online_parameters(), self.target_parameters()
        ):
            target_p.data.mul_(decay).add_(online_p.data, alpha=1 - decay)

    def _encode(
        self,
        video_frames: torch.Tensor,
        use_target: bool = False,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run perception + slot attention.

        Args:
            video_frames: (B, T, C, H, W).
            use_target: If True, use EMA target encoder.
            return_attention: If True, also return slot attention maps.

        Returns:
            slots: (B, T, N_max, d_slot).
            alive: (B, T, N_max).
            attn_maps: (B, T, N_max, M) — only if return_attention=True.
        """
        perception = self.target_perception if use_target else self.perception
        fuser = self.target_fuser if use_target else self.fuser
        slot_attn = self.target_slot_attention if use_target else self.slot_attention

        B, T = video_frames.shape[:2]

        # Perception
        patch_embeddings = perception(video_frames)  # (B, T, N_patches, d_model)
        fused = fuser({"vision": patch_embeddings})  # (B, T, N_patches, d_world)

        # Slot attention per frame
        fused_flat = rearrange(fused, "b t n d -> (b t) n d")
        result = slot_attn(fused_flat, return_attention=return_attention)

        if return_attention:
            slots, alive, attn_maps = result
            attn_maps = rearrange(attn_maps, "(b t) n m -> b t n m", b=B, t=T)
        else:
            slots, alive = result
            attn_maps = None

        slots = rearrange(slots, "(b t) n d -> b t n d", b=B, t=T)
        alive = rearrange(alive, "(b t) n -> b t n", b=B, t=T)

        if return_attention:
            return slots, alive, attn_maps, fused
        return slots, alive, fused

    def forward(
        self,
        video_frames: torch.Tensor,
        actions: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> WorldModelOutput:
        """Full forward pass through all five layers.

        Args:
            video_frames: (B, T, C, H, W) input video.
            actions: (B, T, d_action) optional action sequence.
            targets: (B, T, C, H, W) target frames for loss (usually = input).

        Returns:
            WorldModelOutput with predictions, losses, and diagnostics.
        """
        B, T = video_frames.shape[:2]
        d_slot = self.config.objects.d_slot
        N = self.config.objects.n_max_slots

        # ─── Layer 1+2: Perception + Object Decomposition ───
        encode_result = self._encode(
            video_frames, use_target=False, return_attention=return_attention,
        )
        if return_attention:
            slots, alive, slot_attn_maps, patch_features = encode_result
        else:
            slots, alive, patch_features = encode_result
            slot_attn_maps = None

        # Temporal tracking — build frame-by-frame lists to avoid inplace ops
        hidden = torch.zeros(B, N, d_slot, device=video_frames.device)
        tracked_frames = [slots[:, 0]]
        alive_frames = [alive[:, 0]]

        for t in range(1, T):
            prev_slots = tracked_frames[t - 1]
            curr_slots = slots[:, t]
            prev_alive = alive_frames[t - 1]
            curr_alive = alive[:, t]

            perm, matched = self.slot_tracker(
                prev_slots, curr_slots, prev_alive, curr_alive,
            )
            reordered, hidden = self.slot_tracker.reorder_and_update(
                curr_slots, perm, hidden, matched,
            )
            # Reorder alive mask to match
            perm_1d = perm.long()
            alive_reordered = curr_alive.gather(1, perm_1d)

            tracked_frames.append(reordered)
            alive_frames.append(alive_reordered)

        tracked_slots = torch.stack(tracked_frames, dim=1)  # (B, T, N, d)
        alive = torch.stack(alive_frames, dim=1)  # (B, T, N)

        # Relation graph (GNN)
        slots_flat = rearrange(tracked_slots, "b t n d -> (b t) n d")
        alive_flat = rearrange(alive, "b t n -> (b t) n")
        relational_slots, causal_edges = self.relation_graph(
            slots_flat, alive_flat,
        )
        relational_slots = rearrange(
            relational_slots, "(b t) n d -> b t n d", b=B, t=T,
        )
        edge_weights = rearrange(
            causal_edges, "(b t) n m -> b t n m", b=B, t=T,
        )

        # Causal graph: learned post-hoc from GNN edges (see eval_full.py)

        # ─── Layer 3: Hierarchical Dynamics ───

        # Level 0: SSM — per-object temporal prediction
        obj_temporal = relational_slots.permute(0, 2, 1, 3)  # (B, N, T, d)
        obj_mask = (alive[:, 0] > 0.5).float()  # (B, N)
        level0_out = self.level0_ssm(obj_temporal, obj_mask)
        level0_states = level0_out.permute(0, 2, 1, 3)  # (B, T, N, d)

        # Action conditioning (out-of-place for FSDP compatibility)
        if actions is not None and self.d_action > 0:
            T_act = min(T, actions.shape[1])
            conditioned = []
            for t in range(T):
                if t < T_act:
                    conditioned.append(self.action_conditioner(level0_states[:, t], actions[:, t]))
                else:
                    conditioned.append(level0_states[:, t])
            level0_states = torch.stack(conditioned, dim=1)

        # Event detection
        obj_pooled = (level0_states * alive.unsqueeze(-1)).sum(dim=2)
        n_alive = alive.sum(dim=2, keepdim=True).clamp(min=1.0)
        obj_pooled = obj_pooled / n_alive  # (B, T, d_slot)
        event_scores, event_mask = self.event_detector(obj_pooled)

        # Level 1: Sparse Transformer at events
        event_states, event_pad_mask, K_max = self.hierarchy_manager.gather_events(
            level0_states, event_mask,
        )
        level1_out = self.level1_event(event_states, event_pad_mask, obj_mask)

        # Level 2: Goal dynamics (if enough events)
        level2_out = self.level2_goal(event_states, event_pad_mask)

        # Hierarchy manager: combine all levels
        final_prediction = self.hierarchy_manager(
            level0_states, level1_out, level2_out, event_mask, obj_mask,
        )

        # ─── Layer 4: Causality ───
        causal_losses = self.causal_graph.compute_losses()

        # ─── Loss Computation ───
        losses: dict[str, torch.Tensor] = {}

        if targets is not None:
            # Target encoder (stop-gradient) — also get patch features for SBD
            with torch.no_grad():
                target_slots, target_alive, target_patch_features = self._encode(
                    targets, use_target=True,
                )

            # Combined losses
            losses = self.loss_fn(
                predicted_states=final_prediction,
                target_states=target_slots.detach(),
                object_slots=relational_slots,
                alive_mask=alive,
                frames=video_frames,
                level0_states=level0_states,
                level1_states=level1_out,
                level2_states=level2_out,
                causal_losses={
                    k: v.to(video_frames.device)
                    for k, v in causal_losses.items()
                },
            )

            # Add object consistency losses
            diversity = slot_diversity_loss(
                rearrange(relational_slots, "b t n d -> (b t) n d"),
                rearrange(alive, "b t n -> (b t) n"),
            )
            tracking = slot_tracking_loss(tracked_slots, alive)

            losses["diversity"] = diversity
            losses["tracking"] = tracking

            # ─── Spatial Broadcast Decoder (THE object decomposition signal) ───
            # Apply to EVERY frame for maximum decomposition signal
            sbd_losses = []
            for t_sbd in range(min(T, 4)):  # Up to 4 frames for efficiency
                with torch.amp.autocast("cuda", enabled=False):
                    sbd_loss_t, _, _ = self.spatial_decoder(
                        slots[:, t_sbd].float(),
                        alive[:, t_sbd].float(),
                        target_patch_features[:, t_sbd].float().detach(),
                    )
                if sbd_loss_t.isfinite():
                    sbd_losses.append(sbd_loss_t)
            sbd_loss = torch.stack(sbd_losses).mean() if sbd_losses else torch.tensor(0.0, device=video_frames.device)
            losses["spatial_recon"] = sbd_loss

            # Event detector auxiliary losses
            event_rate_loss = self.event_detector.event_rate_loss(event_scores)
            event_contrastive = self.event_detector.contrastive_event_loss(
                event_scores, obj_pooled,
            )
            losses["event_rate"] = event_rate_loss
            losses["event_contrastive"] = event_contrastive

            # ── Two-stage training schedule (SlotFormer-style) ──
            # Stage 1 (decomposition): SBD dominates, prediction is suppressed
            # Stage 2 (dynamics): prediction restored, SBD becomes regularizer
            #
            # The training_stage attribute is set externally by the trainer
            stage = getattr(self, "_training_stage", 2)  # default to stage 2
            if stage == 1:
                # STAGE 1: Reconstruction-only — slots MUST decompose
                # Build total from scratch (ignore prediction-based total which may be NaN)
                losses["total"] = 5.0 * sbd_loss + 0.1 * diversity
            else:
                # STAGE 2: Full training — prediction + SBD regularizer
                obj_loss = self.config.training.lambda_obj * (diversity + tracking)
                event_loss = 0.05 * (event_rate_loss + event_contrastive)
                spatial_loss = 1.0 * sbd_loss

                if obj_loss.isfinite():
                    losses["total"] = losses["total"] + obj_loss
                if event_loss.isfinite():
                    losses["total"] = losses["total"] + event_loss
                if spatial_loss.isfinite():
                    losses["total"] = losses["total"] + spatial_loss

            # Final NaN guard
            if not losses["total"].isfinite():
                losses["total"] = losses["prediction"]  # Fallback to primary loss

        return WorldModelOutput(
            predicted_states=final_prediction,
            losses=losses,
            object_slots=relational_slots,
            alive_mask=alive,
            event_mask=event_mask,
            event_scores=event_scores,
            causal_graph=self.causal_graph.adjacency.detach().to(video_frames.device),
            abstract_states=level2_out,
            edge_weights=edge_weights,
            slot_attention_maps=slot_attn_maps,
        )
