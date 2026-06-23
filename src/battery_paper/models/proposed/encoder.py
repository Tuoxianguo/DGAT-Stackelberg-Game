"""
Cycle-aware Patching Transformer Encoder.

Input  : (B, N_cycles, F, L_intra)   F = features (V, I, T, Q, ...), L_intra = per-cycle length
Output : (B, N_cycles, D)            cycle-level embeddings  z_t

Design:
  - Each cycle is *one patch*, encoded by a small MLP over (F, L_intra) flattened
    (or 1D conv).  This is the "cycle-aware patching" idea (Bat-T-GNN, 2025).
  - Position embedding over cycle index (learnable + log scale).
  - Stack of TransformerEncoder layers with RoPE-like sinusoid PE.
  - Optional readout token for whole-sequence representation.

Minimal but production-ready (so we can swap PBT/BatteryGPT later).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Encode one cycle  (F, L) → d_model. Uses (channel-wise) 1D conv + GAP + MLP.

    Uses *hard-coded physical normalisation constants* (V/3.6, I/6, T/40, Q/1.2)
    so that the model still sees the absolute capacity-fade signal across cycles.
    Naive InstanceNorm per-cycle erases the cycle-to-cycle drift we want to learn.
    """

    # Physical normalisation factors for [voltage_V, current_A, temperature_C, capacity_Ah]
    # When features have a different layout, the channel-wise constant just becomes a no-op.
    _PHYS_SCALE = [3.6, 6.0, 40.0, 1.2]
    _PHYS_BIAS = [0.0, 0.0, -25.0, 0.0]   # subtract baseline temperature

    def __init__(self, in_features: int, intra_len: int, d_model: int, n_filt: int = 64):
        super().__init__()
        scale = torch.tensor([self._PHYS_SCALE[i % len(self._PHYS_SCALE)]
                              for i in range(in_features)], dtype=torch.float32)
        bias = torch.tensor([self._PHYS_BIAS[i % len(self._PHYS_BIAS)]
                             for i in range(in_features)], dtype=torch.float32)
        self.register_buffer("scale", scale.view(1, in_features, 1))
        self.register_buffer("bias", bias.view(1, in_features, 1))

        self.conv = nn.Sequential(
            nn.Conv1d(in_features, n_filt, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(n_filt),
            nn.Conv1d(n_filt, n_filt, kernel_size=5, padding=2, groups=4),
            nn.GELU(),
            nn.BatchNorm1d(n_filt),
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(n_filt),
            nn.Linear(n_filt, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, F, L) -> reshape to (B*N, F, L)
        B, N, Fc, L = x.shape
        x = x.reshape(B * N, Fc, L)
        x = (x + self.bias) / self.scale     # physical normalisation
        h = self.conv(x)
        h = h.mean(dim=-1)         # GAP across intra-cycle length
        z = self.proj(h)
        return z.view(B, N, -1)


class CyclePositionalEmbed(nn.Module):
    """Sinusoidal PE indexed by cycle number, robust to long sequences."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(1)
        return x + self.pe[:N].unsqueeze(0)


class CycleTransformer(nn.Module):
    def __init__(self, in_features: int, intra_len: int,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 4,
                 dim_ff: int = 256, dropout: float = 0.1, max_cycles: int = 4096):
        super().__init__()
        self.patch = PatchEmbed(in_features, intra_len, d_model)
        self.pe = CyclePositionalEmbed(d_model, max_len=max_cycles)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x    : (B, N, F, L)
        mask : (B, N) {0,1}, 1 = valid
        return z: (B, N, d_model)
        """
        z = self.patch(x)
        z = self.pe(z)
        if mask is not None:
            key_pad_mask = ~mask.bool()
        else:
            key_pad_mask = None
        z = self.enc(z, src_key_padding_mask=key_pad_mask)
        return self.norm(z)
