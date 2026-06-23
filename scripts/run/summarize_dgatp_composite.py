"""Summarize DGAT++ × {HSMM, Graph, Full} composite ablation results.

Reads summary.json from each sweep, computes per-model 3-seed mean ± std
of (5-fold MAPE_mean, 5-fold RMSE_mean), and prints a markdown table that
fits the §4.2.5 ablation table in paper/paper_zh.md.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path("experiments")
# Use the same 3 seeds (42, 7, 2026) for fair comparison vs the composites
COMMON_SEEDS = {42, 7, 2026}
SWEEPS = [
    ("DGAT++ (主干 only)",         "v6_dgat_plus",           "dgat_plus"),
    ("DGAT++ + HSMM",              "sweep_dgatp_hsmm",       "dgat_plus_hsmm"),
    ("DGAT++ + Graph",             "sweep_dgatp_graph",      "dgat_plus_graph"),
    ("DGAT++ + Full (HSMM+Graph)", "sweep_dgatp_full",       "dgat_plus_full"),
]
PARAM_COUNTS = {
    "dgat_plus": 544_000,
    "dgat_plus_hsmm": 575_000,
    "dgat_plus_graph": 807_000,
    "dgat_plus_full": 809_000,
}


def load(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def agg(rows):
    mapes = [r["MAPE_mean"] for r in rows if r.get("MAPE_mean") is not None]
    rmses = [r["RMSE_mean"] for r in rows if r.get("RMSE_mean") is not None]
    if len(mapes) < 1:
        return None, None, None, None
    m_mu = mean(mapes); m_sd = stdev(mapes) if len(mapes) > 1 else 0.0
    r_mu = mean(rmses); r_sd = stdev(rmses) if len(rmses) > 1 else 0.0
    return m_mu, m_sd, r_mu, r_sd


def main():
    print("\n=== DGAT++ × {HSMM, Graph, Full} composite ablation summary ===\n")
    base_mape = None
    rows_out = []
    for label, subdir, model_key in SWEEPS:
        summ = load(ROOT / subdir / "summary.json")
        if summ is None:
            print(f"[SKIP] {subdir}/summary.json not found")
            rows_out.append((label, model_key, None, None, None, None))
            continue
        rows = [r for k, r in summ.items()
                if isinstance(r, dict) and r.get("seed") in COMMON_SEEDS]
        m_mu, m_sd, r_mu, r_sd = agg(rows)
        if m_mu is None:
            print(f"[SKIP] {subdir} has no valid metrics")
            rows_out.append((label, model_key, None, None, None, None))
            continue
        if base_mape is None and model_key == "dgat_plus":
            base_mape = m_mu
        rows_out.append((label, model_key, m_mu, m_sd, r_mu, r_sd))

    print("| # | 模型 | # params | 3-seed mean MAPE (%) | 3-seed mean RMSE | Δ vs DGAT++ |")
    print("|---|---|---|---|---|---|")
    for i, (label, key, m, ms, r, rs) in enumerate(rows_out, 1):
        if m is None:
            print(f"| {i} | {label} | {PARAM_COUNTS.get(key, 0)/1000:.0f} K | _PENDING_ | _PENDING_ | _PENDING_ |")
        else:
            d = f"{m - base_mape:+.2f}" if base_mape is not None else "—"
            print(f"| {i} | {label} | {PARAM_COUNTS.get(key, 0)/1000:.0f} K | "
                  f"{m:.2f} ± {ms:.2f} | {r:.1f} ± {rs:.1f} | {d} |")
    print()


if __name__ == "__main__":
    main()
