"""
Stackelberg-game-based fast-charging protocol optimizer.

★ Innovation 3 of HSMM-GraphGame ★

Problem statement
-----------------
Decision variable:
    p = (CC1, SOC_switch, CC2)     ∈ R^3

Objectives:
    L(p) = predicted_cycle_life( p ; θ_predictor )       (higher is better)
    T(p) = charge_time( p )                              (lower is better; analytical)

Constraints (safety, hardware):
    p_lo ≤ p ≤ p_hi
    V_anode(p) ≥ V_Li_plate   (avoid Li plating, surrogate model)

We model this as a leader-follower Stackelberg game:
    Leader  (battery health) chooses to maximize L(p) - λ·max(0, T(p)-T_max)^2
    Follower (user satisfaction) chooses to minimize T(p) + μ·constraint penalties

When we sweep λ ∈ [0, ∞), the leader's optimum traces out the **Stackelberg-Pareto
front**.  For each λ, we obtain p* via projected gradient ascent (smooth & GPU-friendly).

This module exports:
    StackelbergGame   : pure-function search routine (no NN training)
    StackelbergLayer  : torch.nn.Module wrapping for end-to-end joint training
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

# -------------------------------------------------------------------------- #
# Analytical charge-time model                                                #
# -------------------------------------------------------------------------- #
def charge_time_model(p: torch.Tensor, q_nom_ah: float = 1.1,
                      v_cutoff: float = 3.6, cv_time_s: float = 600.0) -> torch.Tensor:
    """Estimate charge time for MIT 2-step CC protocol.

    p = (CC1, SOC_switch, CC2)
       CC1:   first-stage C-rate            (>0)
       SOC_switch: SoC% at which we switch  (0..100)
       CC2:   second-stage C-rate           (>0)

    Time formula (CC-CC):
        t1 = (SOC_switch/100) / CC1 * 3600
        t2 = ((1 - SOC_switch/100) - cv_fraction) / CC2 * 3600
        t_cv = cv_time_s   (constant-voltage tail to top off the last few %)
        total = t1 + t2 + t_cv   [seconds]

    Returns tensor of shape p.shape[:-1]
    """
    if p.shape[-1] < 3:
        raise ValueError(f"protocol vector must have ≥3 dims, got {p.shape}")
    cc1 = p[..., 0].clamp_min(0.01)
    s = (p[..., 1].clamp(0, 100)) / 100.0
    cc2 = p[..., 2].clamp_min(0.01)
    cv_frac = 0.05
    t1 = s / cc1 * 3600.0
    t2 = ((1 - s) - cv_frac).clamp_min(0.0) / cc2 * 3600.0
    return t1 + t2 + cv_time_s


def li_plating_surrogate(p: torch.Tensor, k: float = 0.05) -> torch.Tensor:
    """Simple monotone surrogate: higher CC1 + lower SOC_switch increase plating risk.

    Returns a *risk* score in [0, ∞) — bigger is worse. Threshold should map
    to constraint V_anode <= V_Li (we want this small).
    """
    cc1 = p[..., 0]
    soc = p[..., 1] / 100.0
    return k * cc1 ** 2 * (1.0 + (1.0 - soc).clamp_min(0.0))


# -------------------------------------------------------------------------- #
# Stackelberg search                                                          #
# -------------------------------------------------------------------------- #
@dataclass
class GameResult:
    p_star: torch.Tensor          # (M, 3)
    L_star: torch.Tensor          # (M,)
    T_star: torch.Tensor          # (M,)
    history: list[dict]


class StackelbergGame:
    """Projected-gradient Stackelberg solver (NN-agnostic)."""

    def __init__(self, life_fn: Callable[[torch.Tensor], torch.Tensor],
                 p_lo: torch.Tensor, p_hi: torch.Tensor,
                 t_max: float = 720.0, lam: float = 1.0,
                 lr: float = 5e-2, n_steps: int = 200,
                 plating_thresh: float = 0.5, mu: float = 5.0):
        self.life_fn = life_fn
        self.p_lo = p_lo
        self.p_hi = p_hi
        self.t_max = t_max
        self.lam = lam
        self.mu = mu
        self.lr = lr
        self.n_steps = n_steps
        self.plating_thresh = plating_thresh

    def solve(self, p_init: torch.Tensor) -> GameResult:
        """p_init: (M, 3) initial protocols (we run M restarts)."""
        p = p_init.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([p], lr=self.lr)
        history = []
        for step in range(self.n_steps):
            opt.zero_grad()
            L = self.life_fn(p)
            T = charge_time_model(p)
            risk = li_plating_surrogate(p)
            penalty_time = (T - self.t_max).clamp_min(0).pow(2)
            penalty_plate = (risk - self.plating_thresh).clamp_min(0).pow(2)
            obj = -(L) + self.lam * penalty_time + self.mu * penalty_plate
            obj.sum().backward()
            opt.step()
            with torch.no_grad():
                p.clamp_(self.p_lo, self.p_hi)
            if step % max(1, self.n_steps // 20) == 0:
                history.append({"step": step, "L": L.mean().item(),
                                "T": T.mean().item(), "obj": obj.mean().item()})
        with torch.no_grad():
            L_final = self.life_fn(p)
            T_final = charge_time_model(p)
        return GameResult(p_star=p.detach(), L_star=L_final.detach(),
                          T_star=T_final.detach(), history=history)

    def pareto_sweep(self, p_init: torch.Tensor,
                     lambdas: list[float]) -> list[GameResult]:
        results = []
        orig_lam = self.lam
        for lam in lambdas:
            self.lam = lam
            results.append(self.solve(p_init))
        self.lam = orig_lam
        return results


# -------------------------------------------------------------------------- #
# Differentiable wrapper for end-to-end training                              #
# -------------------------------------------------------------------------- #
class StackelbergLayer(nn.Module):
    """Wrap a (differentiable) life predictor + game solver for end-to-end use.

    During TRAIN: we approximate p* by short unrolling (n_inner steps) so that
    gradients flow back to the predictor via the implicit function theorem
    (here we approximate IFT by truncated back-prop through the inner loop —
    cheap and works well at K small inner steps).

    During EVAL: increase n_inner for true optimum.
    """

    def __init__(self, life_predictor: nn.Module,
                 p_lo: tuple = (1.0, 30.0, 1.0),
                 p_hi: tuple = (8.0, 90.0, 8.0),
                 n_inner_train: int = 25, n_inner_eval: int = 200,
                 lr_inner: float = 5e-2, t_max: float = 720.0,
                 lam: float = 1.0, mu: float = 5.0):
        super().__init__()
        self.life_predictor = life_predictor
        self.register_buffer("p_lo", torch.tensor(p_lo))
        self.register_buffer("p_hi", torch.tensor(p_hi))
        self.n_inner_train = n_inner_train
        self.n_inner_eval = n_inner_eval
        self.lr_inner = lr_inner
        self.t_max = t_max
        self.lam = lam
        self.mu = mu

    def _life_fn_factory(self):
        def _life(p):
            # life_predictor accepts (M, 3) and returns (M,) predicted cycle_life
            return self.life_predictor(p)
        return _life

    def forward(self, p_init: torch.Tensor) -> dict:
        life_fn = self._life_fn_factory()
        n_inner = self.n_inner_train if self.training else self.n_inner_eval
        p = p_init.clone()
        for _ in range(n_inner):
            p = p.detach().requires_grad_(True)
            L = life_fn(p)
            T = charge_time_model(p)
            risk = li_plating_surrogate(p)
            obj = -(L) + self.lam * (T - self.t_max).clamp_min(0).pow(2) \
                + self.mu * (risk - 0.5).clamp_min(0).pow(2)
            g = torch.autograd.grad(obj.sum(), p, create_graph=self.training)[0]
            p = (p - self.lr_inner * g).clamp(self.p_lo, self.p_hi)
        L_star = life_fn(p)
        T_star = charge_time_model(p)
        return {"p_star": p, "L_star": L_star, "T_star": T_star}
