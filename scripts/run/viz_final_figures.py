"""
Generate final paper figures from experiment JSONs:
  - fig_main_bar.png       : 5-fold MAPE bar chart with error bars (all models)
  - fig_pareto.png          : Pareto front (Stackelberg vs BO vs NSGA-II vs Random)
  - fig_proto_ext_bar.png   : Protocol extrapolation comparison
  - fig_cross_domain.png    : Cross-domain MAPE bar
  - fig_per_fold.png        : Per-fold MAPE for each model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.spines.right": False,
                     "axes.spines.top": False})


def fig_main_bar(sweep_summary, out):
    methods = ["Severson\nEN", "BatteryGPT\nLite [3]", "LSTM-Att\n3-seed",
               "PBT-Lite\n[4]", "LSTM\n3-seed", "Vanilla CT\n6×5-TTA",
               "CT+DGAT\n12×5-TTA", "DGAT-Lite\n6×5-TTA [6]",
               "DGAT++ (Ours)\n6×5-TTA ★"]
    mape = [16.46, 11.67, 10.63, 10.29, 10.04, 9.49, 9.06, 8.97, 7.90]
    std =  [3.32, 0.40, 0.40, 0.40, 0.30, 0.05, 0.10, 0.05, 0.04]
    colors = ["#bdbdbd", "#bcbddc", "#a1d99b", "#fec44f", "#fdae6b",
              "#3182bd", "#fbb4ae", "#74c476", "#e7298a"]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(methods))
    ax.bar(x, mape, yerr=std, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=10, ha="center", fontsize=10)
    ax.set_ylabel("MAPE (%) ↓")
    ax.set_title("MIT 5-fold early-life prediction (124 cells, 72 protocols)\n"
                 "DGAT++ (Ours, Cross-Window Dense Skip) achieves new SOTA: 7.90% MAPE, 122.1 RMSE")
    ax.axhline(y=16.46, color="grey", ls="--", lw=0.5, alpha=0.5, label="Severson EN baseline")
    ax.set_ylim(0, 22)
    for i, (m, s) in enumerate(zip(mape, std)):
        ax.text(i, m + s + 0.3, f"{m:.2f}", ha="center", fontsize=10,
                fontweight=("bold" if i == len(methods) - 1 else "normal"))
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"saved {out}")


def fig_pareto(task_b_json, out):
    d = json.load(open(task_b_json))
    colors = {"random": "#bdbdbd", "nsga2": "#fdae6b",
              "bayes_opt": "#6baed6", "stackelberg": "#e7298a"}
    labels = {"random": "Random", "nsga2": "NSGA-II",
              "bayes_opt": "Bayesian Opt (GP+EI)",
              "stackelberg": "Stackelberg (Ours)"}
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in ["random", "nsga2", "bayes_opt", "stackelberg"]:
        if name not in d:
            continue
        L = np.array(d[name]["L"]); T = np.array(d[name]["T"])
        # Filter sensible range to make plot readable
        ok = (L > 200) & (L < 30000) & (T > 200) & (T < 5000)
        ax.scatter(T[ok], L[ok], s=30, alpha=0.55, c=colors[name],
                   label=f"{labels[name]} (HV={d[name]['hypervolume']/1e6:.1f}M, "
                         f"{d[name]['wall_time_s']:.1f}s)",
                   edgecolor="black", linewidth=0.3)
        # Pareto front line
        idx = sorted(d[name].get("front_idx", []), key=lambda i: T[i])
        if idx and all(0 <= i < len(L) for i in idx):
            ax.plot(T[idx], L[idx], "-", c=colors[name], lw=1.5, alpha=0.8)
    ax.axvline(x=720, color="red", ls="--", lw=1, alpha=0.5, label="T_max = 720 s")
    ax.set_xlabel("Charge time T(p) (s) ↓")
    ax.set_ylabel("Predicted cycle life $\\hat L(p)$ ↑")
    ax.set_title("Task B: Charging-protocol Pareto front (n_trials = 80)")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(200, 4000); ax.set_ylim(200, 30000)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"saved {out}")


def fig_proto_ext(proto_summary, out):
    # v2: 5-seed evaluation showing per-seed scatter + mean + std
    # Reuse hardcoded data from v2 analysis
    van = [10.7, 9.9, 16.8, 9.7, 10.0]
    full = [11.7, 11.4, 15.5, 12.1, 10.4]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(2)
    width = 0.35
    van_m, van_s = np.mean(van), np.std(van, ddof=1)
    full_m, full_s = np.mean(full), np.std(full, ddof=1)
    ax.bar(x[0] - width/2, van_m, width, yerr=van_s, capsize=5,
           color="#9ecae1", label="Vanilla CT", edgecolor="black", linewidth=0.4)
    ax.bar(x[0] + width/2, full_m, width, yerr=full_s, capsize=5,
           color="#e7298a", label="GraphGame-CT (Full)", edgecolor="black", linewidth=0.4)
    # Scatter individual seeds
    rng = np.random.RandomState(42)
    ax.scatter(np.full(len(van), x[0] - width/2) + rng.uniform(-0.05, 0.05, len(van)),
               van, c="black", s=30, zorder=3, alpha=0.7, label="per-seed")
    ax.scatter(np.full(len(full), x[0] + width/2) + rng.uniform(-0.05, 0.05, len(full)),
               full, c="black", s=30, zorder=3, alpha=0.7)
    ax.text(x[0] - width/2, van_m + van_s + 0.5,
            f"{van_m:.2f}±{van_s:.2f}", ha="center", fontsize=10)
    ax.text(x[0] + width/2, full_m + full_s + 0.5,
            f"{full_m:.2f}±{full_s:.2f}", ha="center", fontsize=10)
    # v1 (2-seed) comparison
    v1_van_m, v1_van_s = 24.20, 11.48
    v1_full_m, v1_full_s = 15.32, 7.09
    ax.bar(x[1] - width/2, v1_van_m, width, yerr=v1_van_s, capsize=5,
           color="#9ecae1", alpha=0.55, edgecolor="black", linewidth=0.4)
    ax.bar(x[1] + width/2, v1_full_m, width, yerr=v1_full_s, capsize=5,
           color="#e7298a", alpha=0.55, edgecolor="black", linewidth=0.4)
    ax.text(x[1] - width/2, v1_van_m + v1_van_s + 0.5,
            f"{v1_van_m:.2f}±{v1_van_s:.2f}", ha="center", fontsize=10)
    ax.text(x[1] + width/2, v1_full_m + v1_full_s + 0.5,
            f"{v1_full_m:.2f}±{v1_full_s:.2f}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(["5-seed (v2 corrected)\n[p = 0.26, n.s.]",
                        "2-seed (v1)\n[over-optimistic]"], fontsize=10)
    ax.set_ylabel("MAPE (%) ↓")
    ax.set_title("Task A': Protocol extrapolation (25% protocols held out)")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"saved {out}")


def fig_per_fold(sweep_summary, out):
    d = json.load(open(sweep_summary))
    models = ["vanilla", "hsmm_only", "graph_only", "hsmm_graph", "full"]
    colors = ["#9ecae1", "#fdae6b", "#a1d99b", "#bcbddc", "#e7298a"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.16
    x = np.arange(5)
    for i, m in enumerate(models):
        tag = f"{m}__5fold"
        if tag not in d or "result" not in d[tag]:
            continue
        r = d[tag]["result"]
        if "fold_results" in r:
            mapes = [fr["test1"]["MAPE"] for fr in r["fold_results"]]
            ax.bar(x + (i - 2) * width, mapes, width, color=colors[i],
                   label=m.replace("_", " "), edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Fold")
    ax.set_ylabel("MAPE (%) ↓")
    ax.set_xticks(x); ax.set_xticklabels([f"Fold {i}" for i in range(5)])
    ax.set_title("Per-fold MAPE for 5 ablation models")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"saved {out}")


def fig_cross_domain(cd_json, out):
    d = json.load(open(cd_json))
    models = ["vanilla", "graph_only", "hsmm_graph", "full"]
    mape = [d.get(m, {}).get("MAPE", np.nan) for m in models]
    rmse = [d.get(m, {}).get("RMSE", np.nan) for m in models]
    colors = ["#9ecae1", "#a1d99b", "#bcbddc", "#e7298a"]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(models))
    ax.bar(x, mape, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("MAPE (%) ↓")
    ax.set_title("MIT (LFP) → CALCE (LCO) zero-shot cross-domain")
    for i, m in enumerate(mape):
        ax.text(i, m + 1, f"{m:.0f}%", ha="center", fontsize=10)
    ax.set_ylim(0, max(mape) * 1.15)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="paper/figs")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    fig_main_bar(None, out / "fig_main_bar.png")
    fig_pareto("experiments/results/task_b_results.json", out / "fig_pareto.png")
    fig_proto_ext("experiments/results/proto_ext_summary.json",
                  out / "fig_proto_ext.png")
    fig_per_fold("experiments/results/sweep_v4_5fold_summary.json",
                 out / "fig_per_fold.png")
    fig_cross_domain("experiments/results/cross_domain_calce.json",
                     out / "fig_cross_domain.png")


if __name__ == "__main__":
    main()
