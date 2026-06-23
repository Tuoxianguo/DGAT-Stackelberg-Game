"""Per-cell paired t-test: DGAT++ APE vs DGAT APE (n=124 cells)."""
import json
import numpy as np
from scipy import stats

dgat = json.load(open("experiments/results/v6_dgat_6seed_tta.json"))
plus = json.load(open("experiments/results/v6_dgat_plus_6seed_tta.json"))

# Build {cell_id: ape_median}
def _ape_dict(d):
    return {r["cell_id"]: r["ape_median"] for r in d.get("per_cell", [])}

dgat_ape = _ape_dict(dgat)
plus_ape = _ape_dict(plus)

common = sorted(set(dgat_ape) & set(plus_ape))
print(f"Common cells: {len(common)}")
da = np.array([dgat_ape[c] for c in common])
pa = np.array([plus_ape[c] for c in common])

print(f"DGAT mean APE: {da.mean():.2f}% (median {np.median(da):.2f}%)")
print(f"DGAT++ mean APE: {pa.mean():.2f}% (median {np.median(pa):.2f}%)")
print(f"Mean diff (Plus - DGAT): {(pa - da).mean():+.2f}%")
print(f"n cells where Plus < DGAT: {(pa < da).sum()}/{len(common)} ({(pa<da).mean()*100:.1f}%)")
print(f"n cells where Plus > DGAT: {(pa > da).sum()}/{len(common)} ({(pa>da).mean()*100:.1f}%)")

print(f"\nWilcoxon signed-rank (n={len(common)}):")
w, p = stats.wilcoxon(pa, da, alternative="less")  # H1: Plus has lower APE
sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'
print(f"  W = {w:.1f}, p = {p:.5f} (one-sided: Plus < DGAT) {sig}")

# Two-sided for paper
w2, p2 = stats.wilcoxon(pa, da)
print(f"  two-sided: p = {p2:.5f}")

# Paired t-test
t, pt = stats.ttest_rel(pa, da)
sig_t = '***' if pt<0.001 else '**' if pt<0.01 else '*' if pt<0.05 else 'n.s.'
print(f"\nPaired t-test (n={len(common)}):")
print(f"  t = {t:.3f}, p = {pt:.5f} {sig_t}")
