"""
LSTM baselines for cycle-aware battery life prediction.

Two variants:
  1. LSTMRUL     - vanilla 2-layer BiLSTM + mean pool + MLP head
  2. LSTMAttRUL  - 2-layer BiLSTM + scaled-dot-product attention pool + MLP head

Both share the same Cycle-aware patch embedding as our Vanilla Transformer
(physical normalisation + 1D-Conv + BN + LayerNorm), so the comparison is
strictly "Transformer vs LSTM" at equal capacity / input pipeline.

Residual offset head: log_y = init_log_y_bias + tanh(delta) * scale
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..proposed.encoder import PatchEmbed


class _CyclePosEmbed(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)].unsqueeze(0)


class LSTMRUL(nn.Module):
    """Vanilla BiLSTM RUL predictor."""

    def __init__(self, in_features: int, intra_len: int,
                 d_model: int = 96, n_layers: int = 2, dropout: float = 0.2,
                 init_log_y_bias: float = math.log(800.0),
                 log_y_scale: float = 1.5,
                 d_aux_feat: int = 0):
        super().__init__()
        self.patch = PatchEmbed(in_features, intra_len, d_model)
        self.pe = _CyclePosEmbed(d_model)
        self.lstm = nn.LSTM(d_model, d_model // 2, num_layers=n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
                            bidirectional=True)
        self.norm = nn.LayerNorm(d_model)
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
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.register_buffer("init_log_y_bias", torch.tensor(init_log_y_bias))
        self.log_y_scale = log_y_scale
        with torch.no_grad():
            self.delta_head[-1].weight.zero_()
            self.delta_head[-1].bias.zero_()

    def forward(self, x, mask, feat_aux=None):
        z = self.patch(x)
        z = self.pe(z)
        out, _ = self.lstm(z)                       # (B, N, d_model)
        out = self.norm(out)
        denom = mask.sum(-1, keepdim=True).clamp_min(1.0)
        z_avg = (out * mask.unsqueeze(-1)).sum(1) / denom
        if self.aux_proj is not None:
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


class LSTMAttRUL(nn.Module):
    """BiLSTM + scaled-dot-product attention pool."""

    def __init__(self, in_features: int, intra_len: int,
                 d_model: int = 96, n_layers: int = 2, n_heads: int = 4,
                 dropout: float = 0.2,
                 init_log_y_bias: float = math.log(800.0),
                 log_y_scale: float = 1.5,
                 d_aux_feat: int = 0):
        super().__init__()
        self.patch = PatchEmbed(in_features, intra_len, d_model)
        self.pe = _CyclePosEmbed(d_model)
        self.lstm = nn.LSTM(d_model, d_model // 2, num_layers=n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
                            bidirectional=True)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)
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
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.register_buffer("init_log_y_bias", torch.tensor(init_log_y_bias))
        self.log_y_scale = log_y_scale
        with torch.no_grad():
            self.delta_head[-1].weight.zero_()
            self.delta_head[-1].bias.zero_()

    def forward(self, x, mask, feat_aux=None):
        z = self.patch(x)
        z = self.pe(z)
        out, _ = self.lstm(z)
        out = self.norm(out)
        # Attention pool with learned [CLS] query
        B = out.size(0)
        q = self.cls.expand(B, 1, -1)
        key_pad = ~mask.bool()
        pooled, _ = self.attn(q, out, out, key_padding_mask=key_pad)
        z_avg = pooled.squeeze(1)
        if self.aux_proj is not None:
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
