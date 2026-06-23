"""
Read the sweep summary JSON and produce:
  - a results table (markdown)
  - per-fold MAPE/RMSE figures
  - HSMM stage posterior heatmaps for representative cells (if checkpoints exist)
  - t-SNE of cell embeddings coloured by cycle_life
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def _read_summary(out_root: Path) -> dict:
    sj = out_root / "sweep_summary.json"
    if not sj.exists():
        return {}
    return json.loads(sj.read_text())


def make_table(summary: dict, out_md: Path):
    rows = []
    for tag, info in summary.items():
        if "result" not in info or not isinstance(info["result"], dict):
            rows.append({"tag": tag, "MAPE": "ERR"})
            continue
        r = info["result"]
        if "MAPE_test1_mean" in r:
            row = {"tag": tag,
                   "MAPE_mean": r["MAPE_test1_mean"],
                   "MAPE_std": r["MAPE_test1_std"],
                   "RMSE_mean": r["RMSE_test1_mean"],
                   "RMSE_std": r["RMSE_test1_std"]}
        elif "best" in r:
            row = {"tag": tag,
                   "MAPE_mean": r["best"]["MAPE"],
                   "MAPE_std": 0.0,
                   "RMSE_mean": r["best"]["RMSE"],
                   "RMSE_std": 0.0}
        else:
            row = {"tag": tag, "MAPE_mean": np.nan}
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("MAPE_mean")
    md = ["| Tag | MAPE (%) | RMSE (cycles) |", "|---|---|---|"]
    for _, r in df.iterrows():
        if "MAPE_std" in r:
            md.append(f"| {r['tag']} | {r['MAPE_mean']:.2f} ± {r['MAPE_std']:.2f} | "
                      f"{r['RMSE_mean']:.1f} ± {r['RMSE_std']:.1f} |")
    out_md.write_text("\n".join(md))
    print("\n".join(md))
    df.to_csv(out_md.with_suffix(".csv"), index=False)


def make_per_fold_plot(summary: dict, out_dir: Path):
    """Plot per-fold MAPE bars for each (model x split)."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.8 / max(1, len(summary))
    for i, (tag, info) in enumerate(summary.items()):
        if "result" not in info or not isinstance(info["result"], dict):
            continue
        r = info["result"]
        if "fold_results" in r:
            mapes = [fr["test1"]["MAPE"] for fr in r["fold_results"]]
            xs = np.arange(len(mapes)) + i * width
            ax.bar(xs, mapes, width=width, label=tag.replace("__5fold", ""))
    ax.set_xlabel("Fold")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("5-fold MAPE per model")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "per_fold_mape.png", dpi=150)
    plt.close()
    print(f"saved {out_dir/'per_fold_mape.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="experiments/sweep_v3_5fold")
    args = ap.parse_args()
    out_root = Path(args.out_root)
    summary = _read_summary(out_root)
    if not summary:
        print(f"NO SUMMARY at {out_root}")
        return
    print(f"loaded {len(summary)} runs from {out_root}")
    make_table(summary, out_root / "table.md")
    make_per_fold_plot(summary, out_root)
    print("Done.")


if __name__ == "__main__":
    main()
