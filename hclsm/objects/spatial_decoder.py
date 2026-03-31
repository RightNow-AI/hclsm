"""Spatial Broadcast Decoder for slot-based object decomposition.

Forces each slot to independently reconstruct its spatial region of the image.
Softmax over slots creates competition → objects emerge.

Based on: Locatello et al. (2020) "Object-Centric Learning with Slot Attention"
Enhanced with: DINOSAUR approach (Seitzer et al. 2023) — reconstruct ViT features
instead of raw pixels for better decomposition on real images.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialBroadcastDecoder(nn.Module):
    """Decode each slot independently into a spatial feature map + alpha mask.

    Architecture per slot:
    1. Broadcast slot vector to H_grid x W_grid spatial grid
    2. Add (x, y) positional encoding
    3. CNN decodes to d_target features + 1 alpha channel
    4. Softmax over slots (alive only) → competition for spatial regions
    5. Reconstruction loss: each slot reconstructs its claimed region

    This is THE mechanism that makes slots decompose into objects.
    Without it, slots have no spatial grounding signal.
    """

    def __init__(
        self,
        d_slot: int,
        d_target: int,
        grid_size: int = 14,
        hidden_dim: int = 256,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        self.d_slot = d_slot
        self.d_target = d_target
        self.grid_size = grid_size

        # Positional grid: (x, y) coordinates in [-1, 1]
        ys = torch.linspace(-1, 1, grid_size)
        xs = torch.linspace(-1, 1, grid_size)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        pos_grid = torch.stack([xx, yy], dim=-1)  # (H, W, 2)
        self.register_buffer("pos_grid", pos_grid)

        # CNN: (d_slot + 2) → hidden → ... → (d_target + 1)
        layers = []
        in_ch = d_slot + 2  # slot features + (x, y) position
        for i in range(n_layers):
            out_ch = hidden_dim if i < n_layers - 1 else (d_target + 1)
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            if i < n_layers - 1:
                layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

    def forward(
        self,
        slots: torch.Tensor,
        alive_mask: torch.Tensor,
        target_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode slots and compute spatial reconstruction loss.

        Args:
            slots: (B, N, d_slot) — slot representations.
            alive_mask: (B, N) — alive probabilities.
            target_features: (B, M, d_target) — target patch features
                from frozen encoder. M = grid_size^2.

        Returns:
            recon_loss: scalar reconstruction loss.
            alpha_masks: (B, N, H, W) — per-slot spatial ownership.
            reconstruction: (B, d_target, H, W) — reconstructed features.
        """
        B, N, D = slots.shape
        H = W = self.grid_size

        # 1. Broadcast each slot to spatial grid
        # (B, N, d_slot) → (B*N, d_slot, H, W)
        slots_flat = slots.reshape(B * N, D)
        broadcast = slots_flat[:, :, None, None].expand(B * N, D, H, W)

        # 2. Add positional encoding
        pos = self.pos_grid.permute(2, 0, 1)  # (2, H, W)
        pos = pos.unsqueeze(0).expand(B * N, 2, H, W)
        feat = torch.cat([broadcast, pos], dim=1)  # (B*N, d_slot+2, H, W)

        # 3. CNN decode
        decoded = self.cnn(feat)  # (B*N, d_target+1, H, W)
        decoded = decoded.reshape(B, N, self.d_target + 1, H, W)

        # Split into features and raw alpha
        decoded_feat = decoded[:, :, :self.d_target, :, :]  # (B, N, d_target, H, W)
        raw_alpha = decoded[:, :, self.d_target:, :, :]     # (B, N, 1, H, W)

        # 4. Softmax over ALIVE slots only (dead slots get -inf)
        alive_mask_spatial = alive_mask[:, :, None, None, None]  # (B, N, 1, 1, 1)
        # Set dead slot alphas to -inf so they don't compete
        raw_alpha = raw_alpha + (1 - (alive_mask_spatial > 0.5).float()) * (-1e9)
        alpha = F.softmax(raw_alpha, dim=1)  # (B, N, 1, H, W) — sums to 1 over slots

        # 5. Reconstruction: weighted sum over slots
        reconstruction = (alpha * decoded_feat).sum(dim=1)  # (B, d_target, H, W)

        # 6. Loss: per-slot masked reconstruction
        # Target: (B, M, d_target) → (B, d_target, H, W)
        target = target_features.reshape(B, H, W, self.d_target).permute(0, 3, 1, 2)
        target = target.detach()  # Stop gradient from target encoder

        # Mixture loss: each slot pays for its claimed pixels
        # L = sum_k sum_p [ alpha_{k,p} * ||decoded_{k,p} - target_p||^2 ]
        target_expanded = target.unsqueeze(1).expand_as(decoded_feat)  # (B, N, d_target, H, W)
        per_slot_error = (decoded_feat - target_expanded).pow(2).mean(dim=2)  # (B, N, H, W)
        alpha_squeezed = alpha.squeeze(2)  # (B, N, H, W)

        # Weighted by alpha — slots only pay for pixels they claim
        recon_loss = (alpha_squeezed * per_slot_error).sum(dim=(1, 2, 3)).mean()

        # Return alpha masks for visualization
        alpha_masks = alpha_squeezed.detach()  # (B, N, H, W)

        return recon_loss, alpha_masks, reconstruction
