"""Diagnose which cells the model fails on, by cycle_life bin and protocol."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Load the per-cell predictions from the ensemble JSON
data = json.load(open("experiments/results/v4_ensemble.json"))
records = data["per_cell"]
df = pd.DataFrame(records)

# Add cycle_life bin
df["pred_full_ape"] = np.abs(df["pred_full"] - df["y_true"]) / df["y_true"] * 100
df["pred_graph_ape"] = np.abs(df["pred_graph_only"] - df["y_true"]) / df["y_true"] * 100
df["pred_vanilla_ape"] = np.abs(df["pred_vanilla"] - df["y_true"]) / df["y_true"] * 100
df["pred_ensemble_ape"] = np.abs(df["pred_ensemble"] - df["y_true"]) / df["y_true"] * 100

print(f"loaded {len(df)} cell predictions\n")

print("=== Per-bin APE (Full model) ===")
bins = [0, 300, 600, 900, 1200, 5000]
df["cl_bin"] = pd.cut(df["y_true"], bins=bins, labels=["<300", "300-600",
                                                       "600-900", "900-1200", ">1200"])
agg = df.groupby("cl_bin").agg(
    n=("y_true", "count"),
    ape_full_mean=("pred_full_ape", "mean"),
    ape_graph_mean=("pred_graph_ape", "mean"),
    ape_vanilla_mean=("pred_vanilla_ape", "mean"),
    ape_ensemble_mean=("pred_ensemble_ape", "mean"),
)
print(agg.round(2))

print("\n=== Worst 15 cells (Full model) ===")
worst = df.nlargest(15, "pred_full_ape")[
    ["cell_id", "y_true", "pred_full", "pred_full_ape", "pred_vanilla", "pred_graph_only"]
]
print(worst.to_string(index=False))

print("\n=== Best 15 cells (Full model) ===")
best = df.nsmallest(15, "pred_full_ape")[
    ["cell_id", "y_true", "pred_full", "pred_full_ape"]
]
print(best.to_string(index=False))

print("\n=== Bias analysis: predicted - true ===")
print("  pred_full - true mean:    ", round((df["pred_full"] - df["y_true"]).mean(), 1))
print("  pred_full - true median:  ", round((df["pred_full"] - df["y_true"]).median(), 1))
print("  pred_full - true std:     ", round((df["pred_full"] - df["y_true"]).std(), 1))

# Save for paper
df.to_csv("experiments/results/v4_per_cell_diagnosis.csv", index=False)
print("\nSaved to experiments/results/v4_per_cell_diagnosis.csv")
