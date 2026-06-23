"""Analyze proto extrapolation v2 (5 seeds) + paired t-test."""
import json, statistics as s
from scipy import stats
import numpy as np

d = json.load(open("experiments/results/proto_ext_v2_summary.json"))
agg = {}
for r in d:
    if "best_MAPE" in r:
        agg.setdefault(r["model"], []).append(r["best_MAPE"])

print("Proto-ext (frac=0.25, 5 seeds, paired t-test):")
for m, v in sorted(agg.items()):
    print(f"  {m:8s}  MAPE mean={s.mean(v):.2f} +- std {s.stdev(v):.2f}% "
          f"(n={len(v)}) per-seed: {[f'{x:.1f}' for x in v]}")

van = agg.get("vanilla", [])
ful = agg.get("full", [])
if len(van) == len(ful) and len(van) > 1:
    t, p = stats.ttest_rel(ful, van)
    md = np.mean(np.array(ful) - np.array(van))
    print(f"\nPaired t-test (Full vs Vanilla, n={len(van)}):")
    print(f"  Mean diff (Full - Vanilla): {md:+.2f}%")
    print(f"  t = {t:.3f}, p = {p:.4f}  "
          f"{'***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'}")
else:
    print(f"\nPaired test skipped (n_van={len(van)} vs n_ful={len(ful)})")

# Save summary
out = {
    "vanilla_5seed": {"mapes": van, "mean": float(s.mean(van)) if van else None,
                      "std": float(s.stdev(van)) if len(van) > 1 else None},
    "full_5seed":    {"mapes": ful, "mean": float(s.mean(ful)) if ful else None,
                      "std": float(s.stdev(ful)) if len(ful) > 1 else None},
    "paired_t":      {"t": float(t), "p": float(p),
                      "mean_diff": float(md)} if len(van) == len(ful) and len(van) > 1 else None,
}
json.dump(out, open("experiments/results/proto_ext_v2_analysis.json", "w"), indent=2)
print("\nSaved to experiments/results/proto_ext_v2_analysis.json")
