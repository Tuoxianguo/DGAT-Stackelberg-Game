"""
Simplified reproductions of three 2025 SOTA baselines for fair comparison
on the MIT/Stanford-Toyota dataset (5-fold CV, first-100-cycle setting):

  1. BatteryGPTLite       - GPT-style autoregressive next-cycle reconstruction
                            (causal Transformer + auxiliary recon loss)
                            cf. Chen et al. Nature Comm. 2025 [BatteryGPT]

  2. PBTLite              - Mixture-of-Experts Transformer (4 experts, top-1)
                            cf. PBT arXiv:2512.16334

  3. DGATLite             - Dynamic Graph Attention-Transformer:
                            split 100 cycles into 10 windows-of-10, treat
                            each window as a graph node, attention message
                            passing + Transformer over windows
                            cf. DGAT Energy 2025

All three share the same physical input pipeline (4 channels,
intra_len=64) and residual-offset RUL head as our Vanilla CT, so the
comparison is strictly architectural.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..proposed.encoder import PatchEmbed, CyclePositionalEmbed


# =============================================================================
# 1. BatteryGPTLite
# =============================================================================
class BatteryGPTLite(nn.Module):
    """Causal Transformer + autoregressive next-cycle reconstruction.

    The encoder produces cycle-level embeddings z_t. We attach a small
    decoder head that, given z_{1..t}, predicts ẑ_{t+1}. The reconstruction
    loss is added as auxiliary signal during training. At inference we
    only use the cycle_life head on the mean-pooled embedding.

    This captures BatteryGPT's "autoregressive next-token" inductive bias
    without the full GPT pretraining setup.
    """

    def __init__(self, in_features: int, intra_len: int,
                 d_model: int = 96, n_layers: int = 3, n_heads: int = 4,
                 dropout: float = 0.1,
                 init_log_y_bias: float = math.log(800.0),
                 log_y_scale: float = 1.5, d_aux_feat: int = 0):
        super().__init__()
        self.patch = PatchEmbed(in_features, intra_len, d_model)
        self.pe = CyclePositionalEmbed(d_model)
        # Causal Transformer: same as Vanilla but with causal mask
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.enc_norm = nn.LayerNorm(d_model)
        # Next-cycle reconstruction head (predict z_{t+1} from z_t)
        self.recon_head = nn.Linear(d_model, d_model)
        # RUL residual head
        self.d_aux_feat = d_aux_feat
        if d_aux_feat > 0:
            self.aux_proj = nn.Sequential(
                nn.Linear(d_aux_feat, d_model), nn.GELU(),
                nn.Linear(d_model, d_model))
            d_head_in = d_model * 2
        else:
            self.aux_proj = None
            d_head_in = d_model
        self.delta_head = nn.Sequential(
            nn.LayerNorm(d_head_in),
            nn.Linear(d_head_in, d_model), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1))
        self.register_buffer("init_log_y_bias", torch.tensor(init_log_y_bias))
        self.log_y_scale = log_y_scale
        with torch.no_grad():
            self.delta_head[-1].weight.zero_()
            self.delta_head[-1].bias.zero_()

    def _causal_mask(self, N: int, device) -> torch.Tensor:
        return torch.triu(torch.full((N, N), float("-inf"), device=device),
                          diagonal=1)

    def forward(self, x, mask, feat_aux=None, return_recon: bool = False):
        z_raw = self.patch(x)
        z = self.pe(z_raw)
        causal = self._causal_mask(z.size(1), z.device)
        key_pad = ~mask.bool()
        z_enc = self.enc(z, mask=causal, src_key_padding_mask=key_pad)
        z_enc = self.enc_norm(z_enc)

        # Mean pool for RUL head (valid cycles)
        denom = mask.sum(-1, keepdim=True).clamp_min(1.0)
        z_avg = (z_enc * mask.unsqueeze(-1)).sum(1) / denom
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
        rul = torch.exp(log_y.clamp(2.0, 11.0))
        if return_recon:
            z_pred_next = self.recon_head(z_enc[:, :-1])     # predict z_{t+1}
            return rul, z_pred_next, z_raw[:, 1:]            # pred, target
        return rul


# =============================================================================
# 2. PBTLite - Mixture-of-Experts Transformer
# =============================================================================
class _MoEFFN(nn.Module):
    """Top-1 routing Mixture-of-Experts FFN (replacing Transformer FFN)."""

    def __init__(self, d_model: int, dim_ff: int, n_experts: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_ff), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_ff, d_model))
            for _ in range(n_experts)
        ])
        self.n_experts = n_experts

    def forward(self, x):
        # x: (B, N, d_model)
        gate_logits = self.gate(x)               # (B, N, n_experts)
        weights = F.softmax(gate_logits, dim=-1)  # soft routing (more stable)
        # Compute all experts, weighted sum (dense MoE - more stable than top-1)
        out = torch.zeros_like(x)
        for i, exp in enumerate(self.experts):
            out = out + weights[..., i:i+1] * exp(x)
        return out


class _MoETransformerLayer(nn.Module):
    """Pre-LN Transformer layer with MoE FFN."""

    def __init__(self, d_model: int, n_heads: int, dim_ff: int,
                 n_experts: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.moe = _MoEFFN(d_model, dim_ff, n_experts, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        x_n = self.norm1(x)
        a, _ = self.attn(x_n, x_n, x_n, key_padding_mask=src_key_padding_mask)
        x = x + self.dropout(a)
        x = x + self.dropout(self.moe(self.norm2(x)))
        return x


class PBTLite(nn.Module):
    """Mixture-of-Experts Transformer (simplified PBT)."""

    def __init__(self, in_features: int, intra_len: int,
                 d_model: int = 96, n_layers: int = 3, n_heads: int = 4,
                 n_experts: int = 4, dropout: float = 0.1,
                 init_log_y_bias: float = math.log(800.0),
                 log_y_scale: float = 1.5, d_aux_feat: int = 0):
        super().__init__()
        self.patch = PatchEmbed(in_features, intra_len, d_model)
        self.pe = CyclePositionalEmbed(d_model)
        self.layers = nn.ModuleList([
            _MoETransformerLayer(d_model, n_heads, 4 * d_model, n_experts, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.d_aux_feat = d_aux_feat
        if d_aux_feat > 0:
            self.aux_proj = nn.Sequential(
                nn.Linear(d_aux_feat, d_model), nn.GELU(),
                nn.Linear(d_model, d_model))
            d_head_in = d_model * 2
        else:
            self.aux_proj = None
            d_head_in = d_model
        self.delta_head = nn.Sequential(
            nn.LayerNorm(d_head_in),
            nn.Linear(d_head_in, d_model), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1))
        self.register_buffer("init_log_y_bias", torch.tensor(init_log_y_bias))
        self.log_y_scale = log_y_scale
        with torch.no_grad():
            self.delta_head[-1].weight.zero_()
            self.delta_head[-1].bias.zero_()

    def forward(self, x, mask, feat_aux=None):
        z = self.patch(x)
        z = self.pe(z)
        key_pad = ~mask.bool()
        for layer in self.layers:
            z = layer(z, src_key_padding_mask=key_pad)
        z = self.norm(z)
        denom = mask.sum(-1, keepdim=True).clamp_min(1.0)
        z_avg = (z * mask.unsqueeze(-1)).sum(1) / denom
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


# =============================================================================
# 3. DGATLite - Dynamic Graph Attention-Transformer
# =============================================================================
class DGATLite(nn.Module):
    """
    Split 100 cycles into 10 windows of 10 cycles each. Treat each window
    as a graph node. Inside each node: 1D conv on (10 cycles × F × L). Between
    nodes: dynamic GAT (graph attention between all 10 windows). Final pool
    + RUL head.
    """

    def __init__(self, in_features: int, intra_len: int,
                 d_model: int = 96, n_layers: int = 3, n_heads: int = 4,
                 window_size: int = 10, dropout: float = 0.1,
                 init_log_y_bias: float = math.log(800.0),
                 log_y_scale: float = 1.5, d_aux_feat: int = 0):
        super().__init__()
        self.window_size = window_size
        self.patch = PatchEmbed(in_features, intra_len, d_model)
        # Per-window aggregator: 1D conv over `window_size` cycles → 1 vector
        self.window_aggr = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=window_size,
                      stride=window_size, padding=0),
            nn.GELU(),
            nn.BatchNorm1d(d_model),
        )
        # Inter-window: Dynamic Graph Attention via standard Transformer layers
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.inter_window = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.d_aux_feat = d_aux_feat
        if d_aux_feat > 0:
            self.aux_proj = nn.Sequential(
                nn.Linear(d_aux_feat, d_model), nn.GELU(),
                nn.Linear(d_model, d_model))
            d_head_in = d_model * 2
        else:
            self.aux_proj = None
            d_head_in = d_model
        self.delta_head = nn.Sequential(
            nn.LayerNorm(d_head_in),
            nn.Linear(d_head_in, d_model), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1))
        self.register_buffer("init_log_y_bias", torch.tensor(init_log_y_bias))
        self.log_y_scale = log_y_scale
        with torch.no_grad():
            self.delta_head[-1].weight.zero_()
            self.delta_head[-1].bias.zero_()

    def forward(self, x, mask, feat_aux=None):
        # x: (B, N=100, F, L)
        z = self.patch(x)                              # (B, N, d_model)
        z_t = z.transpose(1, 2)                        # (B, d_model, N)
        w = self.window_aggr(z_t)                      # (B, d_model, N_windows)
        w = w.transpose(1, 2)                          # (B, N_windows, d_model)
        # Build window-level mask: a window is valid if at least 1 cycle valid
        B, N = mask.shape
        n_win = N // self.window_size
        if n_win > 0:
            mask_w = mask[:, :n_win * self.window_size].reshape(B, n_win,
                                                                self.window_size).max(-1).values
        else:
            mask_w = torch.ones(B, w.size(1), device=mask.device)
        key_pad = ~mask_w.bool()
        w = self.inter_window(w, src_key_padding_mask=key_pad)
        w = self.norm(w)
        denom = mask_w.sum(-1, keepdim=True).clamp_min(1.0)
        z_avg = (w * mask_w.unsqueeze(-1)).sum(1) / denom
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


# =============================================================================
# 4. DGATPlusLite - DGAT + Cross-Window Dense Skip Connections (Ours)
# =============================================================================
class _DenseTransformerLayer(nn.Module):
    """Pre-LN Transformer layer that takes a *concatenated* feature stream
    [h_0, h_1, ..., h_{l-1}] (each d_model) and projects back to d_model
    before doing self-attention + FFN. This is the DenseNet-style dense
    connection adapted to the Transformer encoder, applied across windows.

    Each layer's output h_l is appended to the running concat for the next
    layer, so every layer sees ALL previous layer outputs.
    """

    def __init__(self, d_model: int, n_heads: int, dim_ff: int,
                 n_prev: int, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(n_prev * d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, concat_in, key_pad=None):
        x = self.in_proj(concat_in)
        x_n = self.norm1(x)
        a, _ = self.attn(x_n, x_n, x_n, key_padding_mask=key_pad)
        x = x + self.dropout(a)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class DGATPlusLite(nn.Module):
    """
    DGAT-Lite + Cross-Window Dense Skip Connections (our architectural
    improvement). DenseNet-style routing across the inter-window encoder:
    layer l takes a concatenation of {h_0, h_1, ..., h_{l-1}}, projects to
    d_model, then runs attention + FFN.

    Motivation: in 10-window DGAT, far-apart windows (e.g. window 1 and
    window 10) interact only through repeated Transformer layers. Dense
    connections create shortcut paths from h_0 (window-pooled raw features)
    all the way to the final layer, strengthening long-range temporal
    dependencies that signal late-life knee-point degradation.
    """

    def __init__(self, in_features: int, intra_len: int,
                 d_model: int = 96, n_layers: int = 3, n_heads: int = 4,
                 window_size: int = 10, dropout: float = 0.1,
                 init_log_y_bias: float = math.log(800.0),
                 log_y_scale: float = 1.5, d_aux_feat: int = 0):
        super().__init__()
        self.window_size = window_size
        self.d_model = d_model
        self.patch = PatchEmbed(in_features, intra_len, d_model)
        self.window_aggr = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=window_size,
                      stride=window_size, padding=0),
            nn.GELU(),
            nn.BatchNorm1d(d_model),
        )
        self.dense_layers = nn.ModuleList()
        for l in range(n_layers):
            n_prev = l + 1
            self.dense_layers.append(
                _DenseTransformerLayer(d_model, n_heads, 4 * d_model,
                                       n_prev=n_prev, dropout=dropout))
        n_total = n_layers + 1
        self.final_fuse = nn.Sequential(
            nn.LayerNorm(n_total * d_model),
            nn.Linear(n_total * d_model, d_model),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(d_model)
        self.d_aux_feat = d_aux_feat
        if d_aux_feat > 0:
            self.aux_proj = nn.Sequential(
                nn.Linear(d_aux_feat, d_model), nn.GELU(),
                nn.Linear(d_model, d_model))
            d_head_in = d_model * 2
        else:
            self.aux_proj = None
            d_head_in = d_model
        self.delta_head = nn.Sequential(
            nn.LayerNorm(d_head_in),
            nn.Linear(d_head_in, d_model), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1))
        self.register_buffer("init_log_y_bias", torch.tensor(init_log_y_bias))
        self.log_y_scale = log_y_scale
        with torch.no_grad():
            self.delta_head[-1].weight.zero_()
            self.delta_head[-1].bias.zero_()

    def _encode(self, x, mask):
        """Shared encode path. Returns intermediates needed for composite models.

        Returns:
            z_cycle: (B, N, d_model)  patch-embedded per-cycle features
            fused_window: (B, N_win, d_model) DenseNet-fused window embeddings
            cell_emb: (B, d_model) mean-pooled fused windows (cell-level summary)
            mask_w: (B, N_win) window-level validity mask
        """
        z = self.patch(x)                              # (B, N, d_model)
        z_t = z.transpose(1, 2)
        w0 = self.window_aggr(z_t).transpose(1, 2)     # (B, N_win, d_model)
        B, N = mask.shape
        n_win = N // self.window_size
        if n_win > 0:
            mask_w = mask[:, :n_win * self.window_size].reshape(
                B, n_win, self.window_size).max(-1).values
        else:
            mask_w = torch.ones(B, w0.size(1), device=mask.device)
        key_pad = ~mask_w.bool()
        h_stack = [w0]
        for layer in self.dense_layers:
            concat_in = torch.cat(h_stack, dim=-1)
            h_l = layer(concat_in, key_pad=key_pad)
            h_stack.append(h_l)
        fused = self.final_fuse(torch.cat(h_stack, dim=-1))
        fused = self.norm(fused)
        denom = mask_w.sum(-1, keepdim=True).clamp_min(1.0)
        z_avg = (fused * mask_w.unsqueeze(-1)).sum(1) / denom
        return z, fused, z_avg, mask_w

    def forward(self, x, mask, feat_aux=None, return_intermediates: bool = False):
        z_cycle, fused_window, z_avg, mask_w = self._encode(x, mask)
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
        rul = torch.exp(log_y.clamp(2.0, 11.0))
        if return_intermediates:
            return rul, z_cycle, fused_window, z_avg, mask_w
        return rul
