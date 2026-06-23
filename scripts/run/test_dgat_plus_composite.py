"""Smoke test for DGATPlusComposite (HSMM / Graph / Full)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from battery_paper.models.proposed import DGATPlusComposite

for cfg in [("hsmm",  True,  False),
            ("graph", False, True),
            ("full",  True,  True)]:
    name, h, g = cfg
    print(f"=== DGAT++ + {name} ===")
    m = DGATPlusComposite(in_features=4, intra_len=64, d_model=96, n_layers=3,
                          n_heads=4, window_size=10,
                          use_hsmm=h, use_graph=g)
    x = torch.randn(2, 100, 4, 64); mk = torch.ones(2, 100)
    p = torch.tensor([[3.6, 70.0, 4.0], [5.4, 80.0, 3.6]])
    out = m(x, mk, p)
    n_p = sum(pp.numel() for pp in m.parameters())
    print(f"  rul {out.rul_hat.shape}, stage {out.stage_post.shape}, "
          f"cell {out.cell_emb.shape}, n_params {n_p:,}")
    out.rul_hat.sum().backward()
    print("  backward OK\n")
print("ALL OK")
