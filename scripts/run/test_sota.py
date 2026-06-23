"""Smoke test for 3 SOTA baselines."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from battery_paper.models.baselines import BatteryGPTLite, PBTLite, DGATLite

print("Testing BatteryGPTLite...")
m = BatteryGPTLite(in_features=4, intra_len=64, d_model=96, n_layers=3, n_heads=4)
x = torch.randn(2, 100, 4, 64); mk = torch.ones(2, 100)
y = m(x, mk)
print(f"  out shape: {y.shape}, n_params: {sum(p.numel() for p in m.parameters()):,}")
rul, pred, target = m(x, mk, return_recon=True)
print(f"  recon shape: {pred.shape} vs target {target.shape}")
y.sum().backward()
print("  backward OK")

print("\nTesting PBTLite...")
m = PBTLite(in_features=4, intra_len=64, d_model=96, n_layers=3, n_heads=4, n_experts=4)
y = m(x, mk)
print(f"  out shape: {y.shape}, n_params: {sum(p.numel() for p in m.parameters()):,}")
y.sum().backward()
print("  backward OK")

print("\nTesting DGATLite...")
m = DGATLite(in_features=4, intra_len=64, d_model=96, n_layers=3, n_heads=4, window_size=10)
y = m(x, mk)
print(f"  out shape: {y.shape}, n_params: {sum(p.numel() for p in m.parameters()):,}")
y.sum().backward()
print("  backward OK")

print("\nALL OK")
