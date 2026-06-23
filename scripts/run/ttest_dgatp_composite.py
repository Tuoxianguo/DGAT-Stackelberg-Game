"""Paired t-test between DGAT++ composites on per-(seed, fold) MAPE."""
from __future__ import annotations
import json
from pathlib import Path
from scipy import stats
import numpy as np

ROOT = Path("experiments")
COMMON_SEEDS = [42, 7, 2026]


def collect(subdir: str, model_tag: str) -> list[float]:
    out = []
    for s in COMMON_SEEDS:
        seed_dir = ROOT / subdir / f"{model_tag}__seed{s}"
        for fold_dir in sorted(seed_dir.glob("fold_*")):
            rj = fold_dir / "results.json"
            if not rj.exists():
                continue
            try:
                d = json.loads(rj.read_text())
                m = d.get("test1", {}).get("MAPE")
                if m is None or not np.isfinite(m):
                    m = d.get("best", {}).get("MAPE")
                if m is not None:
                    out.append(float(m))
            except Exception:
                pass
    return out


CONFIGS = {
    "DGAT++ (backbone)": ("v6_dgat_plus",      "dgat_plus"),
    "+ HSMM":            ("sweep_dgatp_hsmm",  "dgat_plus_hsmm"),
    "+ Graph":           ("sweep_dgatp_graph", "dgat_plus_graph"),
    "+ Full":            ("sweep_dgatp_full",  "dgat_plus_full"),
}

results = {name: collect(*cfg) for name, cfg in CONFIGS.items()}
for k, v in results.items():
    print(f"{k}: n={len(v):2d}, mean={np.mean(v):.2f}%, std={np.std(v, ddof=1):.2f}%")

base = results["DGAT++ (backbone)"]
print(f"\nPaired t-test vs DGAT++ baseline (n={len(base)} per group, same seeds/folds):")
for k, v in results.items():
    if k == "DGAT++ (backbone)":
        continue
    n = min(len(base), len(v))
    if n < 3:
        print(f"  {k}: insufficient overlap (n={n})")
        continue
    a, b = np.asarray(base[:n]), np.asarray(v[:n])
    t, p = stats.ttest_rel(b, a)  # H1: b > a (worse) corresponds to t>0
    print(f"  {k}: paired t={t:+.3f}, p={p:.4f}  "
          f"(diff mean = {(b - a).mean():+.2f}%)")
