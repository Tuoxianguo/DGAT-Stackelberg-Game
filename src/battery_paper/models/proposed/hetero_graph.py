"""
Protocol-Cell Heterogeneous Graph Neural Network.

★ Innovation 2 of HSMM-GraphGame ★

Node types:
    "protocol"  - one node per (CC1, SOC_switch, CC2) tuple (and other protocol params)
    "cell"      - one node per battery cell

Edge types:
    ("protocol", "trains", "cell")  - which cells were trained under which protocol
    ("cell", "trained_by", "protocol")  - reverse
    ("cell", "similar_to", "cell")  - KNN edge based on embedding cosine similarity

Implementation note:
  - We use a hand-rolled HGT-style message passing because PyG's torch_geometric
    optional C++ ops can be missing on the cloud server.  The implementation is
    O(|E| × d) and works on CPU/GPU.
  - For up to 124 cells x 72 protocols + KNN-5 cell-cell, |E| ≈ 1k, dense MM is fine.

API:
    model = ProtocolCellHGNN(d_p=8, d_c=128, d_hidden=128, n_layers=2)
    out = model(x_p, x_c, edge_pc, edge_cp, edge_cc)
        x_p: (Np, d_p) protocol features
        x_c: (Nc, d_c) cell features (e.g. HSMM-augmented)
        edge_pc: (2, E_pc) [proto_idx; cell_idx]
        edge_cp: (2, E_cp) [cell_idx; proto_idx]
        edge_cc: (2, E_cc) [cell_src; cell_dst]
    returns: dict(protocol=(Np, d_hidden), cell=(Nc, d_hidden))
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeteroAttnConv(nn.Module):
    """Single hetero-aware message passing layer (single relation)."""

    def __init__(self, d_src: int, d_dst: int, d_out: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_out // n_heads
        self.W_q = nn.Linear(d_dst, d_out)
        self.W_k = nn.Linear(d_src, d_out)
        self.W_v = nn.Linear(d_src, d_out)
        self.out = nn.Linear(d_out, d_out)

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor,
                edge_src: torch.Tensor, edge_dst: torch.Tensor) -> torch.Tensor:
        """Aggregate src messages into each dst via softmax attention.

        edge_src, edge_dst: (E,) long
        """
        E = edge_src.size(0)
        if E == 0:
            return torch.zeros(x_dst.size(0), self.n_heads * self.d_head,
                               device=x_dst.device, dtype=x_dst.dtype)
        q = self.W_q(x_dst).view(-1, self.n_heads, self.d_head)
        k = self.W_k(x_src).view(-1, self.n_heads, self.d_head)
        v = self.W_v(x_src).view(-1, self.n_heads, self.d_head)
        q_e = q[edge_dst]          # (E, h, d)
        k_e = k[edge_src]
        v_e = v[edge_src]
        scores = (q_e * k_e).sum(-1) / (self.d_head ** 0.5)   # (E, h)
        # softmax over incoming edges per dst node
        scores_exp = torch.exp(scores - scores.max())
        denom = torch.zeros(x_dst.size(0), self.n_heads, device=x_dst.device, dtype=x_dst.dtype)
        denom.index_add_(0, edge_dst, scores_exp)
        norm = denom[edge_dst].clamp_min(1e-9)
        alpha = scores_exp / norm                              # (E, h)
        msg = v_e * alpha.unsqueeze(-1)                        # (E, h, d)
        agg = torch.zeros(x_dst.size(0), self.n_heads, self.d_head,
                          device=x_dst.device, dtype=x_dst.dtype)
        agg.index_add_(0, edge_dst, msg)
        agg = agg.reshape(-1, self.n_heads * self.d_head)
        return self.out(F.gelu(agg))


class ProtocolCellHGNN(nn.Module):
    def __init__(self, d_p: int, d_c: int, d_hidden: int = 128,
                 n_layers: int = 2, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj_p = nn.Linear(d_p, d_hidden)
        self.proj_c = nn.Linear(d_c, d_hidden)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.ModuleDict({
                "pc": HeteroAttnConv(d_hidden, d_hidden, d_hidden, n_heads),
                "cp": HeteroAttnConv(d_hidden, d_hidden, d_hidden, n_heads),
                "cc": HeteroAttnConv(d_hidden, d_hidden, d_hidden, n_heads),
            })
            self.layers.append(layer)
        self.norm_p = nn.LayerNorm(d_hidden)
        self.norm_c = nn.LayerNorm(d_hidden)
        self.drop = nn.Dropout(dropout)
        self.d_hidden = d_hidden

    def forward(self, x_p: torch.Tensor, x_c: torch.Tensor,
                edge_pc: torch.Tensor,    # (2, E_pc): proto→cell
                edge_cp: torch.Tensor,    # (2, E_cp): cell→proto
                edge_cc: Optional[torch.Tensor] = None) -> dict:
        hp = self.proj_p(x_p)
        hc = self.proj_c(x_c)
        for layer in self.layers:
            # cell update: from protocols + from other cells
            agg_pc = layer["pc"](hp, hc, edge_pc[0], edge_pc[1])
            if edge_cc is not None and edge_cc.numel() > 0:
                agg_cc = layer["cc"](hc, hc, edge_cc[0], edge_cc[1])
                hc_new = hc + self.drop(agg_pc + agg_cc)
            else:
                hc_new = hc + self.drop(agg_pc)
            # protocol update: from its cells
            agg_cp = layer["cp"](hc, hp, edge_cp[0], edge_cp[1])
            hp_new = hp + self.drop(agg_cp)
            hc = self.norm_c(hc_new)
            hp = self.norm_p(hp_new)
        return {"protocol": hp, "cell": hc}


# -------------------------- graph builder helper -------------------------- #
def build_protocol_cell_graph(
    cell_protocols: torch.Tensor,        # (Nc, d_p)  one protocol row per cell
    cell_features: torch.Tensor,         # (Nc, d_c)
    knn_k: int = 5,
) -> dict:
    """Build a tiny graph dict {x_p, x_c, edge_pc, edge_cp, edge_cc, cell_to_proto_idx}.

    Distinct protocols are deduplicated automatically.  Returns tensors on the
    same device as inputs.
    """
    Nc = cell_protocols.size(0)
    # de-duplicate protocols via lex sort of bytes
    proto_id_map = {}
    cell_proto_idx = torch.zeros(Nc, dtype=torch.long, device=cell_protocols.device)
    proto_rows = []
    for i in range(Nc):
        key = tuple(cell_protocols[i].cpu().tolist())
        if key not in proto_id_map:
            proto_id_map[key] = len(proto_id_map)
            proto_rows.append(cell_protocols[i])
        cell_proto_idx[i] = proto_id_map[key]
    Np = len(proto_id_map)
    x_p = torch.stack(proto_rows, dim=0)
    # edges
    proto_arr = cell_proto_idx
    cells_arr = torch.arange(Nc, device=cell_protocols.device)
    edge_pc = torch.stack([proto_arr, cells_arr], dim=0)
    edge_cp = torch.stack([cells_arr, proto_arr], dim=0)
    # cell-cell KNN by cosine of cell features
    if knn_k > 0 and Nc > 1:
        norm = cell_features.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        unit = cell_features / norm
        sim = unit @ unit.t()
        sim.fill_diagonal_(-2.0)
        k = min(knn_k, Nc - 1)
        topk = sim.topk(k, dim=-1).indices              # (Nc, k)
        src = torch.arange(Nc, device=cell_features.device).repeat_interleave(k)
        dst = topk.reshape(-1)
        edge_cc = torch.stack([src, dst], dim=0)
    else:
        edge_cc = torch.zeros(2, 0, dtype=torch.long, device=cell_protocols.device)

    return dict(x_p=x_p, x_c=cell_features, edge_pc=edge_pc,
                edge_cp=edge_cp, edge_cc=edge_cc,
                cell_to_proto_idx=cell_proto_idx)
