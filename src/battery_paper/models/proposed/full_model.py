"""
Full HSMM-GraphGame model that combines:
  Module 1: CycleTransformer encoder
  Module 2: DifferentiableHSMM stage head
  Module 3: ProtocolCellHGNN
  Module 4: StackelbergLayer (used at inference/training of Task B)

We split RUL prediction (Task A) and protocol optimization (Task B) cleanly:
  - For Task A: forward(x, mask, protocol, ...) -> dict(rul_hat, stage_post, ...)
  - For Task B: at inference, instantiate StackelbergLayer with a *protocol-conditioned*
    life predictor head (small MLP on GNN output for protocol nodes) and search p*.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .encoder import CycleTransformer
from .hsmm import DifferentiableHSMM
from .hetero_graph import ProtocolCellHGNN, build_protocol_cell_graph


class _ProtocolConditionedLifeHead(nn.Module):
    """Small MLP that maps protocol embedding -> log10(cycle_life)."""

    def __init__(self, d_p: int, d_hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_p, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.net(p).squeeze(-1)


@dataclass
class FullModelOutput:
    rul_hat: torch.Tensor              # (B,)
    stage_post: torch.Tensor           # (B, T, K)
    log_lik: torch.Tensor              # (B,)
    cell_emb: torch.Tensor             # (B, d_hidden) GNN cell embeddings
    proto_emb: torch.Tensor | None     # (Np, d_hidden) when graph given


class HSMMGraphGameModel(nn.Module):
    """End-to-end model for joint RUL prediction & protocol-aware embedding."""

    def __init__(self,
                 in_features: int,
                 intra_len: int,
                 d_proto: int = 3,
                 d_model: int = 128,
                 n_layers: int = 4,
                 n_heads: int = 4,
                 hsmm_K: int = 4,
                 hsmm_D_max: int = 200,
                 use_graph: bool = True,
                 d_hidden_gnn: int = 128,
                 n_layers_gnn: int = 2,
                 d_aux_feat: int = 0):    # NEW: dim of auxiliary hand-crafted features
        super().__init__()
        self.encoder = CycleTransformer(in_features, intra_len, d_model=d_model,
                                        n_layers=n_layers, n_heads=n_heads)
        self.hsmm = DifferentiableHSMM(K=hsmm_K, D_max=hsmm_D_max, d_z=d_model)
        # cell-level summary projection (mean pool z + HSMM posterior summary)
        self.cell_proj = nn.Sequential(
            nn.LayerNorm(d_model + hsmm_K),
            nn.Linear(d_model + hsmm_K, d_hidden_gnn), nn.GELU(),
        )
        self.use_graph = use_graph
        if use_graph:
            self.gnn = ProtocolCellHGNN(d_p=d_proto, d_c=d_hidden_gnn,
                                        d_hidden=d_hidden_gnn, n_layers=n_layers_gnn,
                                        n_heads=n_heads)
        else:
            self.gnn = None
        # Auxiliary hand-crafted features (Severson v2) → projected to d_hidden_gnn
        self.d_aux_feat = d_aux_feat
        if d_aux_feat > 0:
            self.aux_proj = nn.Sequential(
                nn.Linear(d_aux_feat, d_hidden_gnn), nn.GELU(),
                nn.Linear(d_hidden_gnn, d_hidden_gnn),
            )
            d_head_in = d_hidden_gnn * 2
        else:
            self.aux_proj = None
            d_head_in = d_hidden_gnn
        # Residual-offset RUL head: log_y = init_bias + tanh(delta) * scale
        self.rul_delta = nn.Sequential(
            nn.LayerNorm(d_head_in),
            nn.Linear(d_head_in, d_hidden_gnn), nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_hidden_gnn, 1),
        )
        with torch.no_grad():
            self.rul_delta[-1].weight.zero_()
            self.rul_delta[-1].bias.zero_()
        self.register_buffer("init_log_y_bias", torch.tensor(float(__import__("math").log(800.0))))
        self.log_y_scale = 1.5
        # Backward-compat: provide rul_head attribute (linear shadowing delta)
        self.rul_head = self.rul_delta[-1]

        # auxiliary protocol-conditioned life head (used by Stackelberg layer)
        self.proto_life_head = _ProtocolConditionedLifeHead(d_proto, d_hidden_gnn)

    def forward(self, x: torch.Tensor,                 # (B, N, F, L)
                mask: torch.Tensor,                    # (B, N) {0,1}
                protocols: torch.Tensor,               # (B, d_proto)
                feat_aux: torch.Tensor | None = None,  # (B, d_aux_feat)
                ) -> FullModelOutput:
        z = self.encoder(x, mask=mask)                 # (B, N, d_model)
        hsmm_out = self.hsmm(z, mask=mask)
        # cell summary: mean pool z + last-cycle stage posterior
        denom = mask.sum(-1, keepdim=True).clamp_min(1.0)
        z_avg = (z * mask.unsqueeze(-1)).sum(1) / denom
        last_idx = (mask.sum(-1).long() - 1).clamp_min(0)
        gamma = hsmm_out.posterior_gamma                # (B, T, K)
        last_post = gamma[torch.arange(z.size(0)), last_idx]   # (B, K)
        c_feat_raw = torch.cat([z_avg, last_post], dim=-1)
        c_feat = self.cell_proj(c_feat_raw)             # (B, d_hidden_gnn)

        if self.use_graph and self.gnn is not None:
            graph = build_protocol_cell_graph(protocols, c_feat, knn_k=5)
            out = self.gnn(
                x_p=graph["x_p"], x_c=graph["x_c"],
                edge_pc=graph["edge_pc"], edge_cp=graph["edge_cp"],
                edge_cc=graph["edge_cc"],
            )
            cell_emb = out["cell"]
            proto_emb = out["protocol"]
        else:
            cell_emb = c_feat
            proto_emb = None

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

        return FullModelOutput(rul_hat=rul_hat,
                               stage_post=gamma,
                               log_lik=hsmm_out.log_lik,
                               cell_emb=cell_emb,
                               proto_emb=proto_emb)

    # ------------------------------------------------------------------ #
    # Auxiliary: convert protocol → predicted cycle_life via the         #
    # protocol-conditioned head (used by Stackelberg)                    #
    # ------------------------------------------------------------------ #
    def life_from_protocol(self, p: torch.Tensor) -> torch.Tensor:
        """Predict cycle_life from (M, d_proto) protocols (no encoder needed)."""
        return torch.exp(self.proto_life_head(p))
