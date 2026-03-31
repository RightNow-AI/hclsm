"""Combined loss functions for HCLSM training.

Multi-objective loss: prediction, object consistency, auxiliary decode,
causal regularization, temporal consistency, and SIGReg anti-collapse.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hclsm.config import TrainingConfig


class AuxiliaryDecoder(nn.Module):
    """Lightweight decoder for anti-collapse auxiliary loss.

    Decodes a random 25% crop of the input from the latent state.
    Intentionally weak — just enough to prevent information collapse.
    """

    def __init__(self, d_slot: int, patch_channels: int = 3, crop_size: int = 112) -> None:
        super().__init__()
        self.crop_size = crop_size
        n_output = patch_channels * crop_size * crop_size
        self.decoder = nn.Sequential(
            nn.Linear(d_slot, d_slot * 4),
            nn.ReLU(),
            nn.Linear(d_slot * 4, min(n_output, d_slot * 16)),
        )

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """Decode from pooled slot representations.

        Args:
            slots: (B, d_slot) pooled slot features.

        Returns:
            decoded: (B, d_output) flattened decoded patch.
        """
        return self.decoder(slots)


class CombinedLoss(nn.Module):
    """All HCLSM loss terms combined with configurable weights."""

    def __init__(self, config: TrainingConfig, d_slot: int) -> None:
        super().__init__()
        self.config = config

        # Predictor MLP: maps from dynamics output to target space
        self.predictor = nn.Sequential(
            nn.Linear(d_slot, d_slot * 2),
            nn.ReLU(),
            nn.Linear(d_slot * 2, d_slot),
        )

        # Auxiliary decoder
        self.aux_decoder = AuxiliaryDecoder(d_slot)

        # Compression MLPs for hierarchy consistency
        self.compress_l0_to_l1 = nn.Linear(d_slot, d_slot)
        self.compress_l1_to_l2 = nn.Linear(d_slot, d_slot)

    def prediction_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        alive: torch.Tensor,
    ) -> torch.Tensor:
        """JEPA-style latent prediction loss with stop-gradient target.

        Args:
            predicted: (B, T, N, d_slot) dynamics output.
            target: (B, T, N, d_slot) target encoder output (already detached).
            alive: (B, T, N) alive mask.
        """
        # Predict t+1 from t
        if predicted.shape[1] < 2:
            return torch.tensor(0.0, device=predicted.device)

        pred = self.predictor(predicted[:, :-1])  # (B, T-1, N, d)
        tgt = target[:, 1:]  # (B, T-1, N, d) — stop-grad already applied

        # Use F.mse_loss-style: clamp diff BEFORE squaring to prevent bf16 overflow
        # ** 2 backward (PowBackward0) produces NaN when inputs are large
        residual = pred - tgt
        residual = torch.clamp(residual, -100.0, 100.0)  # Prevent overflow in squaring
        diff = (residual * residual).mean(dim=-1)  # (B, T-1, N) — stable squaring

        return diff.mean()

    def auxiliary_decode_loss(
        self,
        slots: torch.Tensor,
        frames: torch.Tensor,
        alive: torch.Tensor,
    ) -> torch.Tensor:
        """Structural anti-collapse: decode random crop from latent.

        Args:
            slots: (B, T, N, d_slot).
            frames: (B, T, C, H, W) original video frames.
            alive: (B, T, N).
        """
        B, T = slots.shape[:2]

        # Pool slots per frame: (B, T, d_slot)
        mask = (alive > 0.5).float().unsqueeze(-1)  # (B, T, N, 1)
        n_alive = mask.sum(dim=2)  # (B, T, 1)
        n_alive = n_alive.clamp(min=1.0)
        pooled = (slots * mask).sum(dim=2) / n_alive  # (B, T, d_slot)

        # Take first frame for simplicity (Sprint 1)
        pooled_t0 = pooled[:, 0]  # (B, d_slot)
        decoded = self.aux_decoder(pooled_t0)  # (B, d_output)

        # Random crop target
        H, W = frames.shape[-2:]
        cs = self.aux_decoder.crop_size
        if H >= cs and W >= cs:
            h_start = torch.randint(0, H - cs + 1, (1,)).item()
            w_start = torch.randint(0, W - cs + 1, (1,)).item()
            crop = frames[:, 0, :, h_start:h_start + cs, w_start:w_start + cs]
            target_flat = crop.reshape(B, -1)
            # Match output size
            target_flat = target_flat[:, : decoded.shape[-1]]
            decoded = decoded[:, : target_flat.shape[-1]]
            return F.mse_loss(decoded, target_flat.detach())
        else:
            return torch.tensor(0.0, device=slots.device)

    def sigreg_loss(self, slots: torch.Tensor) -> torch.Tensor:
        """SIGReg anti-collapse: match singular value distribution to Gaussian.

        Args:
            slots: (B, T, N, d_slot).
        """
        # Flatten to (n_samples, d_slot)
        Z = slots.reshape(-1, slots.shape[-1])

        if Z.shape[0] < Z.shape[1]:
            return torch.tensor(0.0, device=slots.device)

        # SVD — use a subset for efficiency
        max_samples = min(Z.shape[0], 2048)
        if Z.shape[0] > max_samples:
            idx = torch.randperm(Z.shape[0], device=Z.device)[:max_samples]
            Z = Z[idx]

        # Compute covariance-based proxy instead of SVD (SVD gradients are
        # notoriously unstable and produce NaN at random init)
        # Use variance of embedding norms as a simpler anti-collapse metric
        norms = Z.norm(dim=-1)  # (n_samples,)
        norm_var = norms.var()
        # Penalize low variance (collapse = all same norm)
        target_var = Z.shape[-1] ** 0.5  # expected norm variance for Gaussian
        d = norm_var - target_var
        return (d * d) / (target_var * target_var + 1e-8)

    def hierarchy_consistency_loss(
        self,
        level0_states: torch.Tensor,
        level1_states: torch.Tensor | None,
        level2_states: torch.Tensor | None,
    ) -> torch.Tensor:
        """Ensure coarse levels are consistent compressions of fine levels."""
        loss = torch.tensor(0.0, device=level0_states.device)

        if level1_states is not None and level1_states.shape[0] > 0:
            # Pool L0 over objects and compress
            l0_pooled = level0_states.mean(dim=2)  # (B, T, d)
            l0_compressed = self.compress_l0_to_l1(l0_pooled.mean(dim=1))
            l1_pooled = level1_states.mean(dim=(1, 2))  # (B, d)

            # Clamp to prevent NaN from large SSM outputs at bf16
            l0_compressed = torch.clamp(l0_compressed, -1e4, 1e4)
            l1_pooled = torch.clamp(l1_pooled, -1e4, 1e4)

            l1_target = l1_pooled[:, : l0_compressed.shape[-1]]
            l0_compressed = l0_compressed[:, : l1_target.shape[-1]]

            h_loss = F.mse_loss(l0_compressed, l1_target.detach())
            # Guard against NaN/Inf
            if h_loss.isfinite():
                loss = loss + h_loss

        return loss

    def forward(
        self,
        predicted_states: torch.Tensor,
        target_states: torch.Tensor,
        object_slots: torch.Tensor,
        alive_mask: torch.Tensor,
        frames: torch.Tensor,
        level0_states: torch.Tensor,
        level1_states: torch.Tensor | None,
        level2_states: torch.Tensor | None,
        causal_losses: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all loss terms.

        Returns:
            Dict mapping loss name to scalar tensor.
        """
        losses = {}

        # Primary: latent prediction
        losses["prediction"] = self.prediction_loss(
            predicted_states, target_states, alive_mask,
        )

        # Object consistency (diversity + tracking computed externally)
        # These are added by the model directly

        # Auxiliary decode
        losses["auxiliary"] = self.auxiliary_decode_loss(
            object_slots, frames, alive_mask,
        )

        # SIGReg
        losses["sigreg"] = self.sigreg_loss(object_slots)

        # Hierarchy consistency
        losses["hierarchy"] = self.hierarchy_consistency_loss(
            level0_states, level1_states, level2_states,
        )

        # Causal (from CausalGraph module)
        if causal_losses is not None:
            losses["causal"] = causal_losses.get(
                "causal_total", torch.tensor(0.0, device=frames.device)
            )
        else:
            losses["causal"] = torch.tensor(0.0, device=frames.device)

        # Weighted total
        cfg = self.config
        losses["total"] = (
            cfg.lambda_pred * losses["prediction"]
            + cfg.lambda_aux * losses["auxiliary"]
            + cfg.lambda_sigreg * losses["sigreg"]
            + cfg.lambda_hierarchy * losses["hierarchy"]
            + cfg.lambda_causal * losses["causal"]
        )

        return losses
