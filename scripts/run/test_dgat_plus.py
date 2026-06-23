"""Smoke test for DGATPlusLite."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from battery_paper.models.baselines import DGATPlusLite, DGATLite

print("=== DGATPlusLite ===")
m = DGATPlusLite(in_features=4, intra_len=64, d_model=96, n_layers=3, n_heads=4, window_size=10)
x = torch.randn(2, 100, 4, 64); mk = torch.ones(2, 100)
y = m(x, mk)
n_p = sum(p.numel() for p in m.parameters())
print(f"  out shape: {y.shape}, n_params: {n_p:,}")
y.sum().backward()
print("  backward OK")

print("\n=== DGATLite (for comparison) ===")
m2 = DGATLite(in_features=4, intra_len=64, d_model=96, n_layers=3, n_heads=4, window_size=10)
y2 = m2(x, mk)
n_p2 = sum(p.numel() for p in m2.parameters())
print(f"  out shape: {y2.shape}, n_params: {n_p2:,}")
print(f"\n  Δparams: DGATPlus - DGAT = {n_p - n_p2:+,} ({(n_p-n_p2)/n_p2*100:+.1f}%)")
print("ALL OK")
