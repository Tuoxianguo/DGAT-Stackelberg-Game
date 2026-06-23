"""Quick smoke test for LSTM baselines."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from battery_paper.models.baselines import LSTMRUL, LSTMAttRUL

m = LSTMRUL(4, 64, d_model=96, n_layers=2)
o = m(torch.randn(2, 100, 4, 64), torch.ones(2, 100))
print(f"LSTMRUL OK: out shape {o.shape}, n_params {sum(p.numel() for p in m.parameters()):,}")

m2 = LSTMAttRUL(4, 64, d_model=96, n_layers=2, n_heads=4)
o2 = m2(torch.randn(2, 100, 4, 64), torch.ones(2, 100))
print(f"LSTMAttRUL OK: out shape {o2.shape}, n_params {sum(p.numel() for p in m2.parameters()):,}")

# Backward
o.sum().backward()
o2.sum().backward()
print("backward OK")
