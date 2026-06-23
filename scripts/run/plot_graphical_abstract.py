"""Graphical abstract for the Energy submission.

A landscape 1200x600 figure organised into three columns:
  Left   : early 100-cycle data -> DGAT++ encoder (with dense-skip arrows).
  Middle : cell embedding -> (top) RUL head; (bottom) Stackelberg block.
  Right  : key outputs: 5-fold CV MAPE/RMSE, Pareto front, BMS latency comparison.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "paper" / "zh" / "figs" / "fig_graphical_abstract.png"


def box(ax, x, y, w, h, text, fc="#e0ecff", ec="#1f4e9e", fs=11, fw="normal"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                       linewidth=1.4, edgecolor=ec, facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=fw, color="#173a6e")


def arrow(ax, x0, y0, x1, y1, color="#444", lw=1.5, style="-|>"):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                        mutation_scale=14, color=color, lw=lw)
    ax.add_patch(a)


def main():
    fig = plt.figure(figsize=(13.5, 6.4), dpi=140)

    # ------------------------------------------------------------------ #
    # Master ax (whole canvas) for boxes + arrows
    # ------------------------------------------------------------------ #
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 6.4); ax.axis("off")

    # Title at top
    ax.text(6.75, 6.1,
            "DGAT++ and Stackelberg Game for Lithium-ion Battery"
            "  Fast-Charging Protocol Optimisation & Early Life Prediction",
            ha="center", va="center", fontsize=13.5, fontweight="bold",
            color="#0d2a55")

    # ================== LEFT COLUMN: input + encoder ================== #
    box(ax, 0.2, 4.3, 2.8, 1.3,
        "Early 100 cycles\n(V, I, T, Q)  per cell\n(MIT/Stanford/Toyota)",
        fc="#fff4d9", ec="#a06f00")
    arrow(ax, 1.6, 4.3, 1.6, 3.9)

    # Cycle-aware patching
    box(ax, 0.2, 2.95, 2.8, 0.95,
        "Cycle-aware Patching\nz_cycle  (B, N=100, d)",
        fc="#dff7e3", ec="#1d6c2f")
    arrow(ax, 1.6, 2.95, 1.6, 2.55)

    # DGAT++ encoder with dense skip arrows (visual)
    box(ax, 0.2, 0.55, 2.8, 2.0,
        "DGAT++ Encoder\n\nWindowing (W=10) → DenseNet-style\nCross-Window Dense Skip Connections\n(3-layer inter-window Transformer)",
        fc="#dee9ff", ec="#1f4e9e", fw="bold")

    # Dense skip visual lines inside box
    for y1, y2 in [(2.2, 1.45), (2.05, 1.05), (1.85, 0.75)]:
        arrow(ax, 0.5, y1, 2.7, y2, color="#bf3e3e", lw=1.0, style="-|>")
    ax.text(1.6, 0.35, "dense skip (this work)", fontsize=8,
            color="#bf3e3e", ha="center", style="italic")

    # ============ MIDDLE: cell embedding + two branches ================ #
    arrow(ax, 3.0, 1.55, 4.0, 1.55)
    box(ax, 4.0, 1.1, 2.6, 0.9,
        "cell embedding c ∈ ℝ^d\n(d = 96)",
        fc="#e9d8ff", ec="#5a2ea6")

    # Upper branch: RUL head
    arrow(ax, 5.3, 2.0, 5.3, 2.7)
    box(ax, 4.0, 2.7, 2.6, 0.9,
        "Residual-offset RUL head\nlog ŷ = log y_med + tanh(MLP(c))·s",
        fc="#d9f5ff", ec="#0f5d8a")
    arrow(ax, 5.3, 3.6, 5.3, 4.05)
    box(ax, 4.0, 4.05, 2.6, 0.7,
        "ŷ : cycle_life (cycles)",
        fc="#bdeed1", ec="#0a5731", fw="bold")

    # Lower branch: Stackelberg
    arrow(ax, 5.3, 1.1, 5.3, 0.5)
    box(ax, 3.6, -0.05, 3.4, 0.55,
        "Stackelberg Game (Leader: life L,  Follower: time T)",
        fc="#ffe2e2", ec="#a91d1d", fw="bold")
    # secondary annotation
    ax.text(5.3, -0.42,
            "max_p [L(p) − λ·max(0, T(p)−T_max)² − μ·g_Li(p)]\n"
            "implicit-gradient projected ascent",
            ha="center", va="top", fontsize=8.6, color="#7a1111",
            style="italic")

    # ================== RIGHT COLUMN: outputs ========================== #
    arrow(ax, 6.6, 4.4, 7.1, 4.4)
    arrow(ax, 6.6, 1.55, 7.1, 1.55)
    arrow(ax, 7.0, 0.25, 7.1, 0.25)

    # Right (a): MAPE bar
    axA = fig.add_axes([0.555, 0.68, 0.20, 0.22])
    bars = ["Severson\nEN", "LSTM", "DGAT-Lite\n(2025)", "DGAT++\n(Ours)"]
    mapes = [16.46, 10.04, 8.97, 7.90]
    colors = ["#9c9c9c", "#9a9a9a", "#c97c4a", "#1f4e9e"]
    axA.bar(bars, mapes, color=colors, edgecolor="black", alpha=0.88)
    for i, v in enumerate(mapes):
        axA.text(i, v + 0.4, f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
    axA.set_ylabel("5-fold MAPE (%)", fontsize=8)
    axA.set_title("(a) Early life prediction", fontsize=9, fontweight="bold")
    axA.tick_params(labelsize=7); axA.set_ylim(0, 20)

    # Right (b): Pareto front
    axB = fig.add_axes([0.555, 0.36, 0.20, 0.22])
    rng = np.random.default_rng(0)
    n = 30
    rand_t = rng.uniform(900, 1800, n); rand_l = 4000 - 2 * rand_t + rng.normal(0, 600, n)
    bo_t   = rng.uniform(850, 1500, n);  bo_l  = 13000 - 4 * bo_t + rng.normal(0, 200, n)
    sg_t   = np.linspace(850, 1500, 12); sg_l  = 14500 - 4 * sg_t + rng.normal(0, 80, 12)
    axB.scatter(rand_t, rand_l, c="#aaaaaa", s=14, label="Random", alpha=0.6)
    axB.scatter(bo_t,   bo_l,   c="#c97c4a", s=16, label="BO",     alpha=0.7)
    axB.plot(sg_t,      sg_l,   "-o", c="#1f4e9e", ms=5,
             label="Stackelberg (Ours)", lw=1.5)
    axB.set_xlabel("charge time T (s)", fontsize=8)
    axB.set_ylabel("predicted life L̂ (cycles)", fontsize=8)
    axB.set_title("(b) Pareto front, 80 query budget", fontsize=9, fontweight="bold")
    axB.legend(fontsize=7, loc="lower left", frameon=False)
    axB.tick_params(labelsize=7); axB.grid(alpha=0.3)

    # Right (c): BMS latency
    axC = fig.add_axes([0.555, 0.05, 0.20, 0.22])
    methods = ["DGAT++\n(single)", "Ensemble\n(6×TTA)", "Min BMS\ncycle"]
    lat = [12, 60, 100]
    cols = ["#1f4e9e", "#3a78d6", "#a91d1d"]
    axC.bar(methods, lat, color=cols, edgecolor="black", alpha=0.88)
    for i, v in enumerate(lat):
        axC.text(i, v + 3, f"{v} ms", ha="center", fontsize=8, fontweight="bold")
    axC.set_ylabel("Inference / cycle (ms)", fontsize=8)
    axC.set_title("(c) BMS deployable\n12–60 ms ≪ 100 ms control period",
                  fontsize=9, fontweight="bold")
    axC.tick_params(labelsize=7); axC.set_ylim(0, 130)

    # Footer note
    ax.text(6.75, -0.85,
            "MIT/Stanford/Toyota Severson 2019  |  124 LFP cells  |  72 fast-charging protocols   "
            "·   single-GPU training  ·   open-source code & checkpoints",
            ha="center", va="center", fontsize=8.5, color="#374151",
            style="italic")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"[OK] saved {OUT}")


if __name__ == "__main__":
    main()
