"""Plot DGAT++ × {HSMM, Graph, Full} composite ablation bar chart.

Reads summary.json from each sweep and produces fig_dgatp_composite.png
into the paper/figs directory.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("experiments")
OUT = Path("paper/figs/fig_dgatp_composite.png")
COMMON_SEEDS = {42, 7, 2026}
SWEEPS = [
    ("DGAT++\n(backbone only)",     "v6_dgat_plus",      "dgat_plus",       "#1f77b4"),
    ("DGAT++\n+ HSMM",              "sweep_dgatp_hsmm",  "dgat_plus_hsmm",  "#ff7f0e"),
    ("DGAT++\n+ Graph",             "sweep_dgatp_graph", "dgat_plus_graph", "#2ca02c"),
    ("DGAT++\n+ Full",              "sweep_dgatp_full",  "dgat_plus_full",  "#d62728"),
]


def load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def main():
    labels, mapes, errs, rmses, rmse_errs, colors = [], [], [], [], [], []
    for label, sub, key, color in SWEEPS:
        summ = load(ROOT / sub / "summary.json")
        if summ is None:
            print(f"[SKIP] missing {sub}/summary.json")
            continue
        seeds = [r for k, r in summ.items()
                 if isinstance(r, dict) and r.get("seed") in COMMON_SEEDS
                 and r.get("MAPE_mean") is not None]
        if not seeds:
            print(f"[SKIP] {sub} has no valid seeds")
            continue
        m = [r["MAPE_mean"] for r in seeds]
        rr = [r["RMSE_mean"] for r in seeds]
        labels.append(label)
        mapes.append(mean(m))
        errs.append(stdev(m) if len(m) > 1 else 0.0)
        rmses.append(mean(rr))
        rmse_errs.append(stdev(rr) if len(rr) > 1 else 0.0)
        colors.append(color)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=130)
    x = np.arange(len(labels))

    bars1 = axes[0].bar(x, mapes, yerr=errs, color=colors, edgecolor='black',
                        capsize=4, alpha=0.85)
    axes[0].set_ylabel("MAPE (%) ↓", fontsize=11)
    axes[0].set_title("DGAT++ x {HSMM, Graph, Full} Synergistic Ablation\n"
                      "3-seed x 5-fold CV (seeds = 42, 7, 2026)", fontsize=11)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=10)
    for b, v in zip(bars1, mapes):
        axes[0].text(b.get_x() + b.get_width()/2, v + 0.2, f"{v:.2f}%",
                     ha='center', fontsize=9, fontweight='bold')
    if mapes:
        axes[0].axhline(mapes[0], color='gray', ls='--', alpha=0.5,
                        label=f"DGAT++ baseline = {mapes[0]:.2f}%")
        axes[0].legend(loc='upper left', fontsize=9)

    bars2 = axes[1].bar(x, rmses, yerr=rmse_errs, color=colors, edgecolor='black',
                        capsize=4, alpha=0.85)
    axes[1].set_ylabel("RMSE (cycles) ↓", fontsize=11)
    axes[1].set_title("RMSE synergistic ablation", fontsize=11)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=10)
    for b, v in zip(bars2, rmses):
        axes[1].text(b.get_x() + b.get_width()/2, v + 1, f"{v:.0f}",
                     ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, bbox_inches='tight')
    print(f"[OK] saved {OUT}")


if __name__ == "__main__":
    main()
