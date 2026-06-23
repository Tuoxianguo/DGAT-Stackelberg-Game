"""
Vanilla Transformer baseline (A0 in ablation): encoder + mean pool + MLP.

Re-uses CycleTransformer from the proposed module to keep architecture
comparable; the difference is no HSMM and no graph.

Key implementation detail (fixes collapse on small datasets):
the final Linear bias is initialised to log(median_cycle_life) so the
zero-shot prediction is the dataset median, not 1. This prevents the
"output ≈ 1 ⇒ MAPE ≈ 100%" collapse observed without a good init.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..proposed.encoder import CycleTransformer


class VanillaTransformerRUL(nn.Module):
    """log_y = log_median_bias + tanh(delta_head(z)) * scale

    The tanh ensures bounded predictions (no early-exploding loss) but the
    weights of delta_head start as standard Kaiming init so they can learn
    quickly from the very first epoch.
    """

    def __init__(self, in_features: int, intra_len: int,
                 d_model: int = 128, n_layers: int = 4, n_heads: int = 4,
                 init_log_y_bias: float = math.log(800.0),
                 log_y_scale: float = 1.5,
                 d_aux_feat: int = 0):
        super().__init__()
        self.encoder = CycleTransformer(in_features, intra_len, d_model=d_model,
                                        n_layers=n_layers, n_heads=n_heads)
        self.d_aux_feat = d_aux_feat
        if d_aux_feat > 0:
            self.aux_proj = nn.Sequential(
                nn.Linear(d_aux_feat, d_model), nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            d_head_in = d_model * 2
        else:
            self.aux_proj = None
            d_head_in = d_model
        self.delta_head = nn.Sequential(
            nn.LayerNorm(d_head_in),
            nn.Linear(d_head_in, d_model), nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, 1),
        )
        self.register_buffer("init_log_y_bias", torch.tensor(init_log_y_bias))
        self.log_y_scale = log_y_scale
        with torch.no_grad():
            self.delta_head[-1].weight.zero_()
            self.delta_head[-1].bias.zero_()

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                feat_aux: torch.Tensor | None = None) -> torch.Tensor:
        z = self.encoder(x, mask=mask)
        denom = mask.sum(-1, keepdim=True).clamp_min(1.0)
        z_avg = (z * mask.unsqueeze(-1)).sum(1) / denom
        if self.aux_proj is not None:
            # Always cat (zero vector if feat_aux missing) to keep head shape consistent
            if feat_aux is None:
                feat_aux = torch.zeros(z_avg.size(0), self.d_aux_feat,
                                       device=z_avg.device, dtype=z_avg.dtype)
            aux_h = self.aux_proj(feat_aux)
            head_in = torch.cat([z_avg, aux_h], dim=-1)
        else:
            head_in = z_avg
        delta = torch.tanh(self.delta_head(head_in).squeeze(-1)) * self.log_y_scale
        log_y = self.init_log_y_bias + delta
        return torch.exp(log_y.clamp(2.0, 11.0))
