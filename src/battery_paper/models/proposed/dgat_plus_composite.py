"""
DGAT++ Composite: combines DGATPlusLite encoder with optional HSMM stage head,
Protocol-Cell HGNN, and Game-aux protocol-conditioned life head. This is the
direct extension of HSMMGraphGameModel where the CycleTransformer backbone is
replaced by DGAT++ (DGAT-Lite + Cross-Window Dense Skip Connections).

Output structure compatible with FullModelOutput so it can drop into the
existing training pipeline.

Configurations:
  DGAT++           : encoder only (already covered by DGATPlusLite)
  DGAT++ + HSMM    : compose=`hsmm`  (HSMM operates on cycle-level z, not windows)
  DGAT++ + Graph   : compose=`graph` (HGNN over protocol-cell bipartite + cell-cell KNN)
  DGAT++ + Full    : compose=`full`  (HSMM + Graph + Game-aux)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..baselines.sota_baselines import DGATPlusLite
from .hsmm import DifferentiableHSMM
from .hetero_graph import ProtocolCellHGNN, build_protocol_cell_graph


class _ProtocolConditionedLifeHead(nn.Module):
    def __init__(self, d_p: int, d_hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_p, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, p):
        return self.net(p).squeeze(-1)


@dataclass
class DGATPlusFullOutput:
    rul_hat: torch.Tensor                  # (B,)
    stage_post: torch.Tensor               # (B, T, K)  HSMM posterior (zeros if no HSMM)
    log_lik: torch.Tensor                  # (B,)       HSMM log-likelihood (zeros if off)
    cell_emb: torch.Tensor                 # (B, d_hidden)
    proto_emb: torch.Tensor | None         # (Np, d_hidden) if GNN on


class DGATPlusComposite(nn.Module):
    """DGAT++ backbone + optional HSMM head + optional HGNN + optional Game-aux."""

    def __init__(self,
                 in_features: int, intra_len: int,
                 d_proto: int = 3,
                 d_model: int = 96, n_layers: int = 3, n_heads: int = 4,
                 window_size: int = 10,
                 use_hsmm: bool = False, use_graph: bool = False,
                 hsmm_K: int = 4, hsmm_D_max: int = 200,
                 d_hidden_gnn: int = 96, n_layers_gnn: int = 2,
                 d_aux_feat: int = 0,
                 init_log_y_bias: float = math.log(800.0),
                 log_y_scale: float = 1.5,
                 dropout: float = 0.1):
        super().__init__()
        # DGAT++ encoder backbone
        self.encoder = DGATPlusLite(
            in_features=in_features, intra_len=intra_len,
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            window_size=window_size, dropout=dropout,
            init_log_y_bias=init_log_y_bias, log_y_scale=log_y_scale,
            d_aux_feat=0,
        )

        self.use_hsmm = use_hsmm
        self.use_graph = use_graph

        # HSMM head (operates on per-cycle z, not window-fused, so HSMM still sees
        # full cycle-level dynamics; window-fused goes to GNN)
        if use_hsmm:
            self.hsmm = DifferentiableHSMM(K=hsmm_K, D_max=hsmm_D_max, d_z=d_model)
            cell_in = d_model + hsmm_K
        else:
            self.hsmm = None
            cell_in = d_model
        self.cell_proj = nn.Sequential(
            nn.LayerNorm(cell_in),
            nn.Linear(cell_in, d_hidden_gnn), nn.GELU(),
        )

        # Protocol-Cell HGNN
        if use_graph:
            self.gnn = ProtocolCellHGNN(d_p=d_proto, d_c=d_hidden_gnn,
                                        d_hidden=d_hidden_gnn,
                                        n_layers=n_layers_gnn, n_heads=n_heads)
        else:
            self.gnn = None

        # Auxiliary handcrafted features (kept for API compatibility)
        self.d_aux_feat = d_aux_feat
        if d_aux_feat > 0:
            self.aux_proj = nn.Sequential(
                nn.Linear(d_aux_feat, d_hidden_gnn), nn.GELU(),
                nn.Linear(d_hidden_gnn, d_hidden_gnn))
            d_head_in = d_hidden_gnn * 2
        else:
            self.aux_proj = None
            d_head_in = d_hidden_gnn
        # Residual-offset RUL head
        self.rul_delta = nn.Sequential(
            nn.LayerNorm(d_head_in),
            nn.Linear(d_head_in, d_hidden_gnn), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden_gnn, 1),
        )
        with torch.no_grad():
            self.rul_delta[-1].weight.zero_()
            self.rul_delta[-1].bias.zero_()
        self.register_buffer("init_log_y_bias", torch.tensor(init_log_y_bias))
        self.log_y_scale = log_y_scale
        self.rul_head = self.rul_delta[-1]

        # Game-aux protocol-conditioned head
        self.proto_life_head = _ProtocolConditionedLifeHead(d_proto, d_hidden_gnn)

    def forward(self, x, mask, protocols, feat_aux=None):
        # 1. DGAT++ encoder: z_cycle (B, N, d), fused_window (B, N_win, d),
        #    z_avg (B, d) mean-pooled fused, mask_w
        _, z_cycle, fused_window, z_avg, mask_w = self.encoder(
            x, mask, return_intermediates=True)

        # 2. Optional HSMM on cycle-level z
        if self.hsmm is not None:
            hsmm_out = self.hsmm(z_cycle, mask=mask)
            denom = mask.sum(-1, keepdim=True).clamp_min(1.0)
            z_avg_c = (z_cycle * mask.unsqueeze(-1)).sum(1) / denom
            last_idx = (mask.sum(-1).long() - 1).clamp_min(0)
            gamma = hsmm_out.posterior_gamma
            last_post = gamma[torch.arange(z_cycle.size(0)), last_idx]
            c_feat_raw = torch.cat([z_avg_c, last_post], dim=-1)
            log_lik = hsmm_out.log_lik
        else:
            denom_w = mask_w.sum(-1, keepdim=True).clamp_min(1.0)
            c_feat_raw = (fused_window * mask_w.unsqueeze(-1)).sum(1) / denom_w
            gamma = torch.zeros(z_cycle.size(0), z_cycle.size(1), 1,
                                device=z_cycle.device, dtype=z_cycle.dtype)
            log_lik = torch.zeros(z_cycle.size(0), device=z_cycle.device)
        c_feat = self.cell_proj(c_feat_raw)

        # 3. Optional Protocol-Cell HGNN
        if self.use_graph and self.gnn is not None:
            graph = build_protocol_cell_graph(protocols, c_feat, knn_k=5)
            out = self.gnn(x_p=graph["x_p"], x_c=graph["x_c"],
                           edge_pc=graph["edge_pc"], edge_cp=graph["edge_cp"],
                           edge_cc=graph["edge_cc"])
            cell_emb = out["cell"]
            proto_emb = out["protocol"]
        else:
            cell_emb = c_feat
            proto_emb = None

        # 4. RUL head with residual offset
        if self.aux_proj is not None:
            if feat_aux is None:
                feat_aux = torch.zeros(cell_emb.size(0), self.d_aux_feat,
                                       device=cell_emb.device, dtype=cell_emb.dtype)
            aux_h = self.aux_proj(feat_aux)
            head_in = torch.cat([cell_emb, aux_h], dim=-1)
        else:
            head_in = cell_emb
        delta = torch.tanh(self.rul_delta(head_in).squeeze(-1)) * self.log_y_scale
        log_y = self.init_log_y_bias + delta
        rul_hat = torch.exp(log_y.clamp(2.0, 11.0))

        return DGATPlusFullOutput(rul_hat=rul_hat, stage_post=gamma,
                                  log_lik=log_lik, cell_emb=cell_emb,
                                  proto_emb=proto_emb)

    def life_from_protocol(self, p):
        return torch.exp(self.proto_life_head(p))
