"""
Task B: fast-charging protocol optimisation.

Compares 4 optimisers on the same SURROGATE life model L(p):
  1. Random search
  2. CLO-style Bayesian Optimisation (Gaussian Process + EI)
  3. NSGA-II multi-objective (Pareto baseline)
  4. Stackelberg game (★ our method ★)

The surrogate L(p) is the *protocol-conditioned life head* of a trained
HSMM-GraphGame model (or a simple MLP fitted on (proto, cycle_life) pairs
from training cells if no checkpoint is given).

The cost function is fixed: T(p) computed analytically. Metrics reported:
  - Hyper-volume (HV) of the Pareto front
  - Coverage of the discovered front
  - Best (life, time) point per protocol budget
  - Wall-time per method
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
import torch.nn as nn

from battery_paper.games import StackelbergGame, charge_time_model

# Bounds: (CC1, SOC_switch, CC2)
P_LO = torch.tensor([1.0, 20.0, 1.0])
P_HI = torch.tensor([8.0, 90.0, 8.0])


def _train_surrogate(features_df: pd.DataFrame) -> nn.Module:
    """Train an MLP surrogate L(p) on (proto, cycle_life) pairs."""
    df = features_df.dropna(subset=["cycle_life", "CC1", "SOC_switch", "CC2"]).copy()
    if len(df) < 20:
        raise ValueError(f"too few cells with parsed protocols: {len(df)}")
    X = torch.tensor(df[["CC1", "SOC_switch", "CC2"]].values, dtype=torch.float32)
    y = torch.tensor(np.log(df["cycle_life"].values.astype(np.float32)),
                     dtype=torch.float32)
    net = nn.Sequential(
        nn.Linear(3, 64), nn.GELU(),
        nn.Linear(64, 64), nn.GELU(),
        nn.Linear(64, 1)
    )
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for ep in range(1500):
        opt.zero_grad()
        pred = net(X).squeeze(-1)
        loss = ((pred - y) ** 2).mean()
        loss.backward(); opt.step()
    print(f"[surrogate] trained on {len(df)} cells, final MSE={loss.item():.4f}")
    return net


def _life_from_surrogate(net: nn.Module):
    def f(p: torch.Tensor) -> torch.Tensor:
        return torch.exp(net(p).squeeze(-1))
    return f


# ---------------- Optimisers ---------------- #
def random_search(life_fn, n_trials: int, seed: int = 0):
    rng = np.random.RandomState(seed)
    pts = rng.uniform(P_LO.numpy(), P_HI.numpy(), size=(n_trials, 3))
    pts_t = torch.tensor(pts, dtype=torch.float32)
    with torch.no_grad():
        L = life_fn(pts_t).numpy()
        T = charge_time_model(pts_t).numpy()
    return pts_t, L, T


def bayes_opt(life_fn, n_trials: int, n_init: int = 5, seed: int = 0):
    """Simple BO with GP + EI on - life (maximise life, with time constraint <= 720s)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel
    rng = np.random.RandomState(seed)
    X = rng.uniform(P_LO.numpy(), P_HI.numpy(), size=(n_init, 3))
    def _eval(X_):
        Xt = torch.tensor(X_, dtype=torch.float32)
        with torch.no_grad():
            return life_fn(Xt).numpy(), charge_time_model(Xt).numpy()
    L, T = _eval(X)
    # constraint penalty: subtract penalty for T > 720
    y = L - 5 * np.maximum(0, T - 720)
    for it in range(n_trials - n_init):
        kernel = ConstantKernel(1.0) * Matern(length_scale=[1, 10, 1], nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-3,
                                      normalize_y=True, n_restarts_optimizer=2)
        gp.fit(X, y)
        # sample candidates
        cand = rng.uniform(P_LO.numpy(), P_HI.numpy(), size=(2000, 3))
        mu, sig = gp.predict(cand, return_std=True)
        best = y.max()
        improvement = mu - best
        ei = sig * (improvement / (sig + 1e-9))
        next_x = cand[ei.argmax()]
        X = np.vstack([X, next_x[None]])
        new_L, new_T = _eval(next_x[None])
        new_y = new_L - 5 * np.maximum(0, new_T - 720)
        L = np.concatenate([L, new_L])
        T = np.concatenate([T, new_T])
        y = np.concatenate([y, new_y])
    return torch.tensor(X, dtype=torch.float32), L, T


def nsga2_search(life_fn, n_trials: int, pop_size: int = 16, seed: int = 0):
    """Basic NSGA-II for (max life, min time) with bound box."""
    rng = np.random.RandomState(seed)
    pop = rng.uniform(P_LO.numpy(), P_HI.numpy(), size=(pop_size, 3))
    def _eval(P):
        Pt = torch.tensor(P, dtype=torch.float32)
        with torch.no_grad():
            return life_fn(Pt).numpy(), charge_time_model(Pt).numpy()
    L, T = _eval(pop)
    n_gen = max(1, (n_trials - pop_size) // pop_size)
    for g in range(n_gen):
        # offspring by crossover + gaussian mutation
        idx_p1 = rng.randint(0, pop_size, pop_size)
        idx_p2 = rng.randint(0, pop_size, pop_size)
        alpha = rng.uniform(size=(pop_size, 3))
        off = alpha * pop[idx_p1] + (1 - alpha) * pop[idx_p2]
        off = off + rng.normal(0, 0.2, off.shape) * (P_HI.numpy() - P_LO.numpy())
        off = np.clip(off, P_LO.numpy(), P_HI.numpy())
        L_o, T_o = _eval(off)
        # combine and select by simple Pareto rank + crowding (greedy)
        all_X = np.vstack([pop, off])
        all_L = np.concatenate([L, L_o])
        all_T = np.concatenate([T, T_o])
        # rank: dominated by how many others
        dom_count = np.zeros(len(all_X))
        for i in range(len(all_X)):
            for j in range(len(all_X)):
                if i != j and all_L[j] >= all_L[i] and all_T[j] <= all_T[i] and \
                   (all_L[j] > all_L[i] or all_T[j] < all_T[i]):
                    dom_count[i] += 1
        keep = np.argsort(dom_count)[:pop_size]
        pop = all_X[keep]; L = all_L[keep]; T = all_T[keep]
    return torch.tensor(pop, dtype=torch.float32), L, T


def stackelberg_search(life_fn, n_trials: int, n_lambda: int = 8, seed: int = 0):
    """Sweep λ across the leader's penalty weight to trace Stackelberg-Pareto."""
    rng = np.random.RandomState(seed)
    n_init_per_lambda = max(1, n_trials // n_lambda)
    p_init = torch.tensor(
        rng.uniform(P_LO.numpy(), P_HI.numpy(), size=(n_init_per_lambda, 3)),
        dtype=torch.float32,
    )
    game = StackelbergGame(life_fn, P_LO, P_HI, t_max=720.0, lam=1.0,
                           lr=0.05, n_steps=200)
    lambdas = np.geomspace(1e-3, 1e1, n_lambda)
    all_X, all_L, all_T = [], [], []
    for lam in lambdas:
        game.lam = float(lam)
        res = game.solve(p_init)
        all_X.append(res.p_star)
        all_L.append(res.L_star.detach().numpy())
        all_T.append(res.T_star.detach().numpy())
    return (torch.cat(all_X, 0),
            np.concatenate(all_L), np.concatenate(all_T))


# ---------------- Metrics ---------------- #
def _pareto_front(L: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Indices on the (max-L, min-T) Pareto front."""
    order = np.argsort(-L)
    front = []
    best_t = float("inf")
    for idx in order:
        if T[idx] < best_t:
            front.append(idx)
            best_t = T[idx]
    return np.asarray(front)


def hypervolume(L: np.ndarray, T: np.ndarray, ref_L: float, ref_T: float) -> float:
    """HV with reference (ref_L_low, ref_T_high). All points with L > ref_L and T < ref_T."""
    mask = (L > ref_L) & (T < ref_T)
    if mask.sum() == 0:
        return 0.0
    front = _pareto_front(L[mask], T[mask])
    L_f = L[mask][front]; T_f = T[mask][front]
    # sort by L ascending
    order = np.argsort(L_f)
    L_f = L_f[order]; T_f = T_f[order]
    hv = 0.0
    prev_T = ref_T
    for li, ti in zip(L_f, T_f):
        hv += (li - ref_L) * (prev_T - ti)
        prev_T = ti
    return float(hv)


# ---------------- Main ---------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", default="data/interim/severson_features.csv")
    ap.add_argument("--manifest_csv", default="data/interim/mit_meta.csv")
    ap.add_argument("--n_trials", type=int, default=64)
    ap.add_argument("--out_dir", default="experiments/task_b")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    # Build features_df with parsed protocol values
    meta = pd.read_csv(args.manifest_csv)
    from battery_paper.data.bsebench_loader import parse_policy
    rows = []
    for _, r in meta.iterrows():
        pp = parse_policy(r["policy_readable"])
        rows.append({"cell_id": r["cell_id"], "cycle_life": r["cycle_life"],
                     "CC1": pp.get("CC1"), "SOC_switch": pp.get("SOC_switch"),
                     "CC2": pp.get("CC2")})
    feats = pd.DataFrame(rows)
    print(f"loaded {len(feats)} cells with protocols")
    net = _train_surrogate(feats)
    life_fn = _life_from_surrogate(net)

    results = {}
    for name, opt_fn in [
        ("random", random_search),
        ("bayes_opt", bayes_opt),
        ("nsga2", nsga2_search),
        ("stackelberg", stackelberg_search),
    ]:
        t0 = time.time()
        X, L, T = opt_fn(life_fn, n_trials=args.n_trials)
        wall = time.time() - t0
        feasible = T <= 1500
        L_f = L[feasible]; T_f = T[feasible]
        if len(L_f) == 0:
            best_L = float(np.nanmax(L)); best_T = float(np.nanmin(T))
        else:
            idx = np.argmax(L_f - 0.05 * T_f / 3600.0)
            best_L = float(L_f[idx]); best_T = float(T_f[idx])
        hv = hypervolume(L, T, ref_L=200.0, ref_T=3600.0)
        front_idx = _pareto_front(L, T)
        tight = T <= 720
        best_L_tight = float(L[tight].max()) if tight.sum() else float("nan")
        print(f"  {name:12s}  hv={hv:9.1f}  best_L={best_L:6.0f}  best_T={best_T:5.0f}s  "
              f"L|T<=720={best_L_tight:6.0f}  |front|={len(front_idx):2d}  "
              f"wall={wall:5.1f}s")
        results[name] = {
            "n_trials": int(args.n_trials),
            "hypervolume": hv,
            "best_L_under_constraint": best_L,
            "best_T_under_constraint": best_T,
            "best_L_tight_720s": best_L_tight,
            "n_front": int(len(front_idx)),
            "wall_time_s": wall,
            "X": X.tolist(),
            "L": L.tolist(),
            "T": T.tolist(),
            "front_idx": front_idx.tolist(),
        }
    with open(Path(args.out_dir) / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved to", args.out_dir)


if __name__ == "__main__":
    main()
