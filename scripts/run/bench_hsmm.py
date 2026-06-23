"""Time the HSMM forward to ensure vectorisation paid off."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from battery_paper.models.proposed import DifferentiableHSMM

device = "cuda" if torch.cuda.is_available() else "cpu"
B, T, K, D, d_z = 16, 100, 4, 200, 96
print(f"device={device} B={B} T={T} K={K} D={D} d_z={d_z}")

hsmm = DifferentiableHSMM(K=K, D_max=D, d_z=d_z).to(device)
z = torch.randn(B, T, d_z, device=device, requires_grad=True)
m = torch.ones(B, T, device=device)

# Warmup
for _ in range(3):
    out = hsmm(z, m)
    out.log_lik.sum().backward()
torch.cuda.synchronize() if device == "cuda" else None

n = 10
t0 = time.time()
for _ in range(n):
    out = hsmm(z, m)
    (-out.log_lik.mean()).backward()
torch.cuda.synchronize() if device == "cuda" else None
elapsed = time.time() - t0
print(f"  forward+backward x {n}: {elapsed:.2f}s   per call: {elapsed/n*1000:.1f} ms")
print(f"  log_lik mean: {out.log_lik.mean().item():.2f}")
print(f"  gamma shape: {tuple(out.posterior_gamma.shape)}")
