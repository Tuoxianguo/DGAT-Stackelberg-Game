"""
Differentiable Explicit-Duration Hidden Semi-Markov Model (EDHMM).

★ Innovation 1 of HSMM-GraphGame ★

Standard EDHMM forward recursion (Yu 2010 §IV-A, Murphy 2002 §2.2.1):

  α_t(j) = log p(z_{1:t}, segment ends at t in state j)
         = logsumexp_{d=1..min(D, t)} {
             log p(d | j)
             + Σ_{u=t-d+1..t} log p(z_u | j)
             + logsumexp_{i ≠ j} α_{t-d}(i) + log A(i, j)        if t > d
             + log π(j)                                            if t = d
           }

Final:
  log p(z_{1:T}) = logsumexp_j α_T(j)

Implementation:
- We pre-compute cumulative emission log-likelihoods (B, T+1, K) for O(1) run
  log-prob queries: Σ_{u=t-d+1..t} log p(z_u | j) = csum[t] - csum[t-d].
- Loop over t = 1..T and d = 1..min(D, t).  Total cost O(T * D * K²).
- We work entirely in log-space with NEG_INF = -1e9 sentinel (NOT -inf, to
  keep gradients finite).

Posterior γ_t(j) = P(s_t = j | z_{1:T}) is computed by combining α (forward
that ends segments at each t) with a backward β, but for simplicity we use
the *segment-end* posterior:
  γ̃_t(j) = softmax_j α_t(j)
(this is the prob that a segment of state j ends at exactly t, which is
sufficient for our stage-aware RUL head).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = -1e9


def _log_gauss_diag(z: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """z: (..., D), mu/log_var: (K, D). Returns (..., K) of log N(z; mu_k, diag(exp(log_var_k)))."""
    z_exp = z.unsqueeze(-2)
    mu = mu.view(*([1] * (z.dim() - 1)), *mu.shape)
    lv = log_var.view(*([1] * (z.dim() - 1)), *log_var.shape)
    d = (z_exp - mu) ** 2
    out = -0.5 * (d * torch.exp(-lv) + lv + math.log(2 * math.pi))
    return out.sum(-1)


@dataclass
class HSMMOutput:
    log_lik: torch.Tensor          # (B,)
    posterior_gamma: torch.Tensor  # (B, T, K) segment-end posterior
    rul_hat: torch.Tensor          # (B,)
    duration_logits: torch.Tensor  # (K, D_max)


class DifferentiableHSMM(nn.Module):
    def __init__(self, K: int = 4, D_max: int = 200, d_z: int = 128,
                 left_to_right: bool = True, weibull_init: bool = True,
                 min_var: float = 1e-2, max_emit_norm: float = 50.0) -> None:
        super().__init__()
        self.K = K
        self.D_max = D_max
        self.d_z = d_z
        self.left_to_right = left_to_right
        self.min_var = min_var
        self.max_emit_norm = max_emit_norm

        # Emission: per-state mean and log-variance, diagonal Gaussian.
        # The emissions get a *layer-norm* on z BEFORE the gaussian to keep
        # the squared distance in a sane range.
        self.input_norm = nn.LayerNorm(d_z, elementwise_affine=False)
        self.mu = nn.Parameter(torch.randn(K, d_z) * 0.1)
        # initial log-variance ≈ log(1.0); will be clamped to [min_var, 100]
        self.log_var = nn.Parameter(torch.zeros(K, d_z))

        # Initial distribution: prefer state 0
        init = torch.full((K,), -3.0)
        init[0] = 0.0
        self.log_pi = nn.Parameter(init)

        # Transition matrix (left-to-right enforced if requested)
        if left_to_right:
            A_init = torch.full((K, K), float(NEG_INF))
            for i in range(K - 1):
                A_init[i, i + 1] = 0.0
            A_init[K - 1, K - 1] = 0.0
            # Not trainable to preserve structure
            self.A_logits = nn.Parameter(A_init, requires_grad=False)
        else:
            self.A_logits = nn.Parameter(torch.randn(K, K) * 0.01)

        # Duration distribution per state (Weibull-init categorical)
        d_logits = torch.zeros(K, D_max)
        if weibull_init:
            for k in range(K):
                shape = 1.5
                scale = max(5.0, (k + 1) * (D_max / (K + 1)))
                ts = torch.arange(1, D_max + 1, dtype=torch.float32)
                pdf = (shape / scale) * (ts / scale).pow(shape - 1) * \
                    torch.exp(-(ts / scale).pow(shape))
                d_logits[k] = torch.log(pdf + 1e-8)
        self.duration_logits = nn.Parameter(d_logits)

    # ------------------------------------------------------------------ #
    def _log_A(self) -> torch.Tensor:
        return F.log_softmax(self.A_logits, dim=-1)

    def _log_pi(self) -> torch.Tensor:
        return F.log_softmax(self.log_pi, dim=0)

    def _log_duration(self) -> torch.Tensor:
        return F.log_softmax(self.duration_logits, dim=-1)

    def _emission_loglik(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, T, D) -> (B, T, K). Apply layer-norm to z first for stability."""
        z_n = self.input_norm(z)
        log_var = self.log_var.clamp(math.log(self.min_var), 4.0)
        out = _log_gauss_diag(z_n, self.mu, log_var)
        return out.clamp(min=-self.max_emit_norm, max=0.0)

    # ------------------------------------------------------------------ #
    def forward(self, z: torch.Tensor, mask: Optional[torch.Tensor] = None,
                return_posterior: bool = True) -> HSMMOutput:
        B, T, _ = z.shape
        device, dtype = z.device, z.dtype
        if mask is None:
            mask = torch.ones(B, T, dtype=dtype, device=device)

        log_emit = self._emission_loglik(z)              # (B, T, K)
        log_emit = log_emit * mask.unsqueeze(-1)
        # Cumulative sum: csum[b, t+1, k] = sum_{u<=t} log_emit[b, u, k]
        csum = torch.zeros(B, T + 1, self.K, device=device, dtype=dtype)
        csum[:, 1:] = torch.cumsum(log_emit, dim=1)

        log_A = self._log_A()                            # (K_prev, K_next)
        log_pi = self._log_pi()                          # (K,)
        log_d = self._log_duration()                     # (K, D_max)
        D = min(self.D_max, T)

        # α[t, j] = log p(z_{1:t}, segment ends at t in state j)
        # Stored as (B, T+1, K). α[0] = NEG_INF.
        alpha = torch.full((B, T + 1, self.K), NEG_INF,
                           device=device, dtype=dtype)

        # Vectorised forward.
        #
        # We avoid the inner d-loop by precomputing a "transitioned alpha":
        #   alpha_in[t, k] = log Σ_i exp(α[t, i] + log A(i, k))   for t in [0, T]
        # alpha_in[0, k] is undefined; we don't use it (replaced by π for d=t case).
        #
        # Then for each t and d in 1..d_max_t:
        #   contrib_in(t, d, k) = alpha_in[t-d, k]   if d < t
        #                       = log π(k)            if d == t
        # We construct a "starter" table starter[s, k] (s ranges 0..T) where:
        #   starter[s, k] = log π(k)        if s == 0  (run starts at cycle 1 → t-d=0)
        #                 = alpha_in[s, k]  if s >= 1
        # Then for a given t, d ∈ 1..min(D, t):
        #   contrib[t, d, k] = starter[t-d, k] + emit_run[t, d, k] + log_d[k, d-1]
        # and alpha[t, k] = logsumexp_d contrib[t, d, k]
        for t in range(1, T + 1):
            d_max_t = min(D, t)
            # alpha_in: (B, d_max_t, K)
            # build starter table for the relevant prefix
            starter_indices = torch.arange(t - d_max_t, t, device=device)  # length d_max_t
            # but we need a STARTER table where index 0 -> π and index s>=1 -> alpha_in[s-1+1]
            # Simpler: directly compute each d's starter:
            #   if t - d == 0  → π
            #   else           → alpha_in[t-d]
            # gather alpha_in_for_d  shape (B, d_max_t, K):
            #   alpha_in_for_d[b, d-1, k] = (alpha_in[t-d, k] if t-d>=1 else log_pi[k])
            # We compute alpha_in for prev range t-d_max_t..t-1 first.
            # Pre-shift: alpha[t-d_max_t..t-1] shape (B, d_max_t, K)
            prev_alpha = alpha[:, t - d_max_t:t, :].flip(1)  # idx 0 -> d=1 (closest)
            # alpha_in: for each entry of prev_alpha, do logsumexp_i over log_A
            ain = torch.logsumexp(
                prev_alpha.unsqueeze(-1) + log_A.unsqueeze(0).unsqueeze(0),
                dim=2,
            )  # (B, d_max_t, K)
            # Replace the d == t entry (last entry in the flipped axis, corresponds to t-d=0)
            # with log_pi (broadcasting)
            if t <= D:
                # the entry at axis index t-1 corresponds to d = t  (since flip ordering)
                ain = ain.clone()
                ain[:, t - 1, :] = log_pi.unsqueeze(0)
            # Emission runs (also vectorised)
            cs_t = csum[:, t, :].unsqueeze(1)               # (B, 1, K)
            cs_prev = csum[:, t - d_max_t:t, :].flip(1)     # (B, d, K)
            emit_runs = cs_t - cs_prev                      # (B, d, K)
            log_d_slice = log_d[:, :d_max_t].t().unsqueeze(0)  # (1, d, K)
            contrib = ain + emit_runs + log_d_slice         # (B, d, K)
            alpha[:, t, :] = torch.logsumexp(contrib, dim=1)

        # Pad-aware log-likelihood: take α at the last valid t.
        Tval = mask.sum(dim=1).long().clamp(min=1)
        idx = Tval
        log_lik = alpha[torch.arange(B, device=device), idx, :]
        log_lik = torch.logsumexp(log_lik, dim=-1)

        # Posterior: segment-end posterior γ̃_t(k) = softmax over k of α[t, k].
        alpha_t = alpha[:, 1:T + 1, :]                                  # (B, T, K)
        denom = torch.logsumexp(alpha_t, dim=-1, keepdim=True)
        gamma = torch.exp((alpha_t - denom).clamp(min=-50, max=0))
        gamma = gamma * mask.unsqueeze(-1)

        # Expected remaining cycle life given current stage at last valid t
        last_gamma = gamma[torch.arange(B, device=device), (idx - 1).clamp(min=0), :]
        d_grid = torch.arange(1, self.D_max + 1, device=device, dtype=dtype)
        E_d = (F.softmax(self.duration_logits, dim=-1) * d_grid).sum(-1)
        k_idx = torch.arange(self.K, device=device, dtype=dtype)
        remaining_stages = (self.K - k_idx - 0.5).clamp(min=0.0)
        rul_per_state = E_d * remaining_stages
        rul_hat = (last_gamma * rul_per_state).sum(-1)

        return HSMMOutput(log_lik=log_lik, posterior_gamma=gamma,
                          rul_hat=rul_hat, duration_logits=self.duration_logits)

    @torch.no_grad()
    def stage_assign(self, z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward(z, mask).posterior_gamma.argmax(-1)
