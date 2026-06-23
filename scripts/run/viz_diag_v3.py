"""Diagnostic visualisations for v3 (6-seed × 5-TTA) ensemble."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.spines.right": False,
                     "axes.spines.top": False})

df = pd.read_csv("experiments/results/v3_per_cell_diag.csv")
df["err"] = df["pred"] - df["y_true"]

# Figure: 2x2 diagnostic
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

# (1) predicted vs true scatter
ax = axes[0, 0]
ax.scatter(df["y_true"], df["pred"], s=35, c="#3182bd", alpha=0.65,
           edgecolors="black", linewidths=0.4, label=f"n={len(df)}")
lim = [0, 2000]
ax.plot(lim, lim, "k--", lw=1, alpha=0.5, label="y = x")
# ±15% bounds
ax.fill_between(lim, [l * 0.85 for l in lim], [l * 1.15 for l in lim],
                color="grey", alpha=0.10, label="±15% band")
ax.set_xlim(0, 2000); ax.set_ylim(0, 2000)
ax.set_aspect("equal")
ax.set_xlabel("True cycle life")
ax.set_ylabel("Predicted cycle life")
ax.set_title(f"(a) Pred vs True (MAPE = 9.48%, RMSE = 137.7)")
ax.legend(fontsize=9, loc="lower right")

# (2) Per-bin MAPE bar
ax = axes[0, 1]
bins_label = ["<300\n(n=2)", "300-500\n(n=31)", "500-700\n(n=21)",
              "700-900\n(n=34)", "900-1200\n(n=28)", ">1200\n(n=8)"]
mape_bin = [112.0, 4.5, 10.9, 7.1, 6.1, 21.1]
colors_b = ["#e7298a" if v > 15 else "#3182bd" for v in mape_bin]
xs = np.arange(len(bins_label))
ax.bar(xs, mape_bin, color=colors_b, edgecolor="black", linewidth=0.4)
for i, m in enumerate(mape_bin):
    ax.text(i, m + 3, f"{m:.1f}%", ha="center", fontsize=10,
            fontweight=("bold" if m > 15 else "normal"))
ax.set_xticks(xs); ax.set_xticklabels(bins_label, fontsize=9)
ax.set_ylabel("MAPE (%)")
ax.set_title("(b) MAPE by cycle-life bin: middle range strong, tails weak")
ax.axhline(y=9.48, color="grey", ls="--", lw=0.6, alpha=0.6, label="overall 9.48%")
ax.legend(fontsize=9)
ax.set_ylim(0, 125)

# (3) Error histogram
ax = axes[1, 0]
ax.hist(df["err"], bins=30, color="#3182bd", edgecolor="black",
        linewidth=0.4, alpha=0.85)
ax.axvline(x=0, color="black", lw=1)
ax.axvline(x=df["err"].mean(), color="#e7298a", lw=1.5,
           label=f"mean = {df['err'].mean():+.0f}")
ax.axvline(x=df["err"].median(), color="orange", lw=1.5, ls="--",
           label=f"median = {df['err'].median():+.0f}")
ax.set_xlabel("Pred − True (cycles)")
ax.set_ylabel("Cell count")
ax.set_title(f"(c) Residual distribution (bias = −25 cycles, std = 136)")
ax.legend(fontsize=9)

# (4) APE cumulative distribution (how many cells under each error level)
ax = axes[1, 1]
sorted_ape = np.sort(df["ape"].values)
cdf = np.arange(1, len(sorted_ape) + 1) / len(sorted_ape) * 100
ax.plot(sorted_ape, cdf, c="#3182bd", lw=2)
thresholds = [5, 10, 15, 20, 30]
for t in thresholds:
    pct = (sorted_ape <= t).sum() / len(sorted_ape) * 100
    ax.axvline(x=t, color="grey", ls=":", lw=0.5)
    ax.text(t + 0.5, pct + 2, f"{pct:.0f}% cells\nunder {t}% APE",
            fontsize=8, color="#555")
ax.set_xscale("log")
ax.set_xlim(0.1, 200)
ax.set_ylim(0, 105)
ax.set_xlabel("Absolute Percentage Error (%)")
ax.set_ylabel("Cumulative fraction of cells (%)")
ax.set_title("(d) APE CDF — 50% cells under 5% APE")
ax.grid(True, alpha=0.3, ls=":")

plt.tight_layout()
plt.savefig("paper/figs/fig_diagnostic_v3.png", dpi=150)
plt.close()
print("Saved paper/figs/fig_diagnostic_v3.png")
