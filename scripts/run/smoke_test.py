"""
Quick smoke test: verifies imports, builds tiny tensors, runs one forward pass
of every component. No real data required.

Run on cloud:
    python -m scripts.run.smoke_test
"""

from __future__ import annotations

import sys
import time
import traceback

import numpy as np


def _ok(msg: str) -> None:
    print(f"  OK   {msg}")


def _bad(msg: str, e: Exception | None = None) -> None:
    print(f"  FAIL {msg}: {e}")
    if e is not None:
        traceback.print_exc()


def main() -> int:
    fails = 0
    print("[smoke] imports...")
    try:
        import torch
        _ok(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
    except Exception as e:
        _bad("torch", e); fails += 1; return 1
    try:
        from battery_paper.models.proposed import (
            DifferentiableHSMM, CycleTransformer, ProtocolCellHGNN,
            HSMMGraphGameModel)
        from battery_paper.models.baselines import VanillaTransformerRUL
        from battery_paper.games import StackelbergGame, charge_time_model
        from battery_paper.features import compute_dq_variance
        _ok("battery_paper imports")
    except Exception as e:
        _bad("battery_paper imports", e); fails += 1; return 1

    print("[smoke] HSMM forward...")
    try:
        K, D, T, B = 4, 50, 30, 2
        hsmm = DifferentiableHSMM(K=K, D_max=D, d_z=16)
        z = torch.randn(B, T, 16)
        m = torch.ones(B, T)
        out = hsmm(z, m)
        assert out.log_lik.shape == (B,)
        assert out.posterior_gamma.shape == (B, T, K)
        _ok(f"HSMM log_lik={out.log_lik.tolist()} rul={out.rul_hat.tolist()}")
    except Exception as e:
        _bad("HSMM", e); fails += 1

    print("[smoke] Transformer encoder...")
    try:
        F, L, N = 5, 32, 20
        enc = CycleTransformer(in_features=F, intra_len=L, d_model=16,
                               n_layers=2, n_heads=2)
        x = torch.randn(B, N, F, L)
        mask = torch.ones(B, N)
        z = enc(x, mask)
        assert z.shape == (B, N, 16)
        _ok(f"Encoder z={tuple(z.shape)}")
    except Exception as e:
        _bad("encoder", e); fails += 1

    print("[smoke] HGNN forward...")
    try:
        gnn = ProtocolCellHGNN(d_p=3, d_c=16, d_hidden=16, n_layers=1)
        Nc = 8; Np = 3
        x_c = torch.randn(Nc, 16)
        x_p = torch.randn(Np, 3)
        proto_idx = torch.tensor([0, 0, 1, 1, 1, 2, 2, 0])
        edge_pc = torch.stack([proto_idx, torch.arange(Nc)])
        edge_cp = torch.stack([torch.arange(Nc), proto_idx])
        edge_cc = torch.tensor([[0, 1, 2, 3, 4], [1, 0, 3, 2, 5]])
        out = gnn(x_p, x_c, edge_pc, edge_cp, edge_cc)
        assert out["cell"].shape == (Nc, 16)
        _ok(f"HGNN cell={tuple(out['cell'].shape)} proto={tuple(out['protocol'].shape)}")
    except Exception as e:
        _bad("HGNN", e); fails += 1

    print("[smoke] Full HSMMGraphGameModel forward + backward...")
    try:
        model = HSMMGraphGameModel(in_features=F, intra_len=L,
                                   d_model=16, n_layers=2, n_heads=2,
                                   hsmm_K=3, hsmm_D_max=30, use_graph=True,
                                   d_hidden_gnn=16, n_layers_gnn=1)
        x = torch.randn(B, N, F, L)
        m = torch.ones(B, N)
        proto = torch.tensor([[3.6, 70.0, 4.0], [5.4, 80.0, 3.6]])
        out = model(x, m, proto)
        y = torch.tensor([500.0, 800.0])
        loss = torch.nn.functional.mse_loss(torch.log(out.rul_hat.clamp_min(1)),
                                            torch.log(y)) - out.log_lik.mean() * 0.01
        loss.backward()
        gnorm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        _ok(f"FullModel loss={loss.item():.3f} grad_norm={gnorm:.2f}")
    except Exception as e:
        _bad("FullModel", e); fails += 1

    print("[smoke] StackelbergGame Pareto sweep...")
    try:
        def life_fn(p):
            cc1 = p[..., 0]; soc = p[..., 1] / 100.0; cc2 = p[..., 2]
            # toy: longer life with low currents and moderate switch
            return 1500 - 150 * (cc1 + cc2) + 200 * (1 - (soc - 0.6) ** 2)
        p_lo = torch.tensor([1.0, 20.0, 1.0])
        p_hi = torch.tensor([8.0, 90.0, 8.0])
        game = StackelbergGame(life_fn, p_lo, p_hi, t_max=720.0, lam=1.0,
                               lr=0.05, n_steps=80)
        p_init = torch.tensor([[4.0, 60.0, 4.0], [5.0, 70.0, 5.0]])
        res = game.solve(p_init)
        _ok(f"Stackelberg p*={res.p_star.tolist()}, L={res.L_star.tolist()}, T={res.T_star.tolist()}")
        sweep = game.pareto_sweep(p_init, lambdas=[0.0, 0.5, 1.0, 5.0])
        _ok(f"Sweep L: {[r.L_star.mean().item() for r in sweep]}, T: {[r.T_star.mean().item() for r in sweep]}")
    except Exception as e:
        _bad("Stackelberg", e); fails += 1

    print()
    if fails == 0:
        print("[smoke] ALL OK")
    else:
        print(f"[smoke] {fails} FAILURES")
    return fails


if __name__ == "__main__":
    sys.exit(main())
