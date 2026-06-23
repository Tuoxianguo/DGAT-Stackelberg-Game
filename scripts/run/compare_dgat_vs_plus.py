"""Paired t-test: DGATPlusLite vs DGATLite on 6-seed × 5-fold MAPE.

Reads per-fold MAPE from sweep_summary.json files of both runs.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import stats


def _per_fold_mapes(summary_dir: Path, seeds: list[int]) -> list[float]:
    """Returns flat list of per-fold MAPE across all seeds."""
    out = []
    for seed in seeds:
        run = summary_dir / f"dgat__seed{seed}" / "results.json"
        run_plus = summary_dir / f"dgat_plus__seed{seed}" / "results.json"
        # Actually, per-fold MAPE is in fold_X/results.json
        for fi in range(5):
            for tag, base in [("dgat", run.parent.parent), ("dgat_plus", run_plus.parent.parent)]:
                fold_json = base / f"{tag}__seed{seed}" / f"fold_{fi}" / "results.json"
                if fold_json.exists():
                    d = json.loads(fold_json.read_text())
                    out.append((tag, seed, fi, d["best"]["MAPE"]))
    return out


def main():
    # Read sweep_summary jsons
    dgat_root = Path("experiments/results/v6_dgat_summary.json")
    plus_root = Path("experiments/results/v6_dgat_plus_summary.json")
    if not dgat_root.exists() or not plus_root.exists():
        # Fallback: use the bundled sweep summary
        print("Per-fold summaries not available locally; will fetch from cloud.")
        return
    d_dgat = json.loads(dgat_root.read_text())
    d_plus = json.loads(plus_root.read_text())
    # Extract per-seed FINAL 5-fold MAPE
    dgat_mapes, plus_mapes = [], []
    for k, v in d_dgat.items():
        if "result" in v and "MAPE_test1_mean" in v["result"]:
            dgat_mapes.append(v["result"]["MAPE_test1_mean"])
    for k, v in d_plus.items():
        if "result" in v and "MAPE_test1_mean" in v["result"]:
            plus_mapes.append(v["result"]["MAPE_test1_mean"])
    print(f"DGAT      mean MAPE (per-seed): {dgat_mapes}")
    print(f"DGATPlus  mean MAPE (per-seed): {plus_mapes}")
    if len(dgat_mapes) == len(plus_mapes) and len(dgat_mapes) > 1:
        t, p = stats.ttest_rel(plus_mapes, dgat_mapes)
        md = np.mean(np.array(plus_mapes) - np.array(dgat_mapes))
        print(f"\nPaired t-test (Plus vs DGAT, n={len(dgat_mapes)} seeds):")
        print(f"  Mean diff: {md:+.2f}%")
        print(f"  t = {t:.3f}, p = {p:.4f}")


if __name__ == "__main__":
    main()
