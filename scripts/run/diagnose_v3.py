"""Deep diagnostic on the v3 (6-seed × 5-TTA) ensemble: per-bin, worst cells, bias."""
import json
import numpy as np
import pandas as pd

d = json.load(open("experiments/results/v6_6seed_tta_pc.json"))
df = pd.DataFrame(d["per_cell"])
df["pred"] = df["median"]
df["ape"] = df["ape_median"]
df["err"] = df["pred"] - df["y_true"]

# Load v4 sweep result for HSMM-GraphGame full predictions for comparison
v4 = pd.read_csv("experiments/results/v4_per_cell_diagnosis.csv")
# Merge by cell_id
m = df.merge(v4[["cell_id", "pred_full", "pred_vanilla", "pred_graph_only",
                 "pred_full_ape", "pred_vanilla_ape", "pred_graph_ape"]],
             on="cell_id", how="left")

print(f"N = {len(df)} cells")
print(f"\n=== Aggregate metrics ===")
print(f"  MAPE     :  {df['ape'].mean():.2f}% (median of seeds)")
print(f"  Median APE: {df['ape'].median():.2f}%")
print(f"  RMSE     :  {np.sqrt((df['err']**2).mean()):.1f} cycles")
print(f"  MAE      :  {df['err'].abs().mean():.1f} cycles")
print(f"  Bias (pred-true): {df['err'].mean():+.1f} cycles "
      f"(std {df['err'].std():.1f})")

print(f"\n=== Per cycle_life bin ===")
bins = [0, 300, 500, 700, 900, 1200, 5000]
df["cl_bin"] = pd.cut(df["y_true"], bins=bins,
                     labels=["<300", "300-500", "500-700", "700-900", "900-1200", ">1200"])
agg = df.groupby("cl_bin").agg(
    n=("y_true", "count"),
    y_med=("y_true", "median"),
    pred_med=("pred", "median"),
    ape_mean=("ape", "mean"),
    ape_max=("ape", "max"),
    bias_mean=("err", "mean"),
)
print(agg.round(1).to_string())

print(f"\n=== Worst 10 cells (v3 6-seed × 5-TTA median) ===")
worst = df.nlargest(10, "ape")[["cell_id", "y_true", "pred", "ape"]]
print(worst.to_string(index=False))

print(f"\n=== Best 10 cells ===")
best = df.nsmallest(10, "ape")[["cell_id", "y_true", "pred", "ape"]]
print(best.to_string(index=False))

print(f"\n=== Comparison: v3 (6-seed TTA) vs v4 single Full ===")
print(f"  v3 (median ens.) MAPE: {m['ape'].mean():.2f}%")
print(f"  v4 Full (single) MAPE: {m['pred_full_ape'].mean():.2f}%")
print(f"  v4 Graph (single)MAPE: {m['pred_graph_ape'].mean():.2f}%")
print(f"  v4 Vanilla(single)MAPE:{m['pred_vanilla_ape'].mean():.2f}%")

print(f"\n=== Where v3 helps most (vs v4 Full) ===")
m["improve_vs_full"] = m["pred_full_ape"] - m["ape"]
biggest_help = m.nlargest(10, "improve_vs_full")[
    ["cell_id", "y_true", "pred_full", "pred", "improve_vs_full"]
]
print(biggest_help.to_string(index=False))

print(f"\n=== Where v3 hurts (vs v4 Full) ===")
biggest_hurt = m.nsmallest(10, "improve_vs_full")[
    ["cell_id", "y_true", "pred_full", "pred", "improve_vs_full"]
]
print(biggest_hurt.to_string(index=False))

# Bias correction analysis
print(f"\n=== Bias by predicted bin (over/under-prediction) ===")
df["pred_bin"] = pd.cut(df["pred"], bins=bins,
                       labels=["<300", "300-500", "500-700", "700-900", "900-1200", ">1200"])
print(df.groupby("pred_bin").agg(
    n=("y_true","count"),
    bias=("err","mean"),
    bias_med=("err","median"),
).round(1).to_string())

# Save diagnostic for figure use
df.to_csv("experiments/results/v3_per_cell_diag.csv", index=False)
print("\nSaved experiments/results/v3_per_cell_diag.csv")
