"""Paired t-test: DGAT++ vs DGAT (6 seeds × 5-fold per-seed mean MAPE)."""
import numpy as np
from scipy import stats

# Per-seed FINAL 5-fold MAPE (from sweep logs)
DGAT = {       42: 13.15,    7: 12.54, 2026: 14.10,  1024: 15.10,  100: 14.23,  777: 11.95}
DGATPLUS = {  42: 13.75,    7: 13.94, 2026: 10.72,  100: 11.98,    777: 11.77,  1024: 10.91}

seeds = sorted(set(DGAT) & set(DGATPLUS))
d = np.array([DGAT[s] for s in seeds])
dp = np.array([DGATPLUS[s] for s in seeds])
print(f"Seeds: {seeds}")
print(f"DGAT      per-seed MAPE: {d.tolist()}")
print(f"DGAT++    per-seed MAPE: {dp.tolist()}")
print(f"\nMean diff (DGAT++ - DGAT): {(dp - d).mean():+.2f}%")
print(f"Median diff: {np.median(dp - d):+.2f}%")
t, p = stats.ttest_rel(dp, d)
sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'
print(f"Paired t-test (Plus - DGAT): t = {t:.3f}, p = {p:.4f} {sig}")
print(f"DGAT      mean: {d.mean():.2f} +- {d.std(ddof=1):.2f}")
print(f"DGAT++    mean: {dp.mean():.2f} +- {dp.std(ddof=1):.2f}")

print(f"\n=== 6-seed × 5-TTA Median ensemble ===")
print(f"  DGAT:      8.97% MAPE / 136.6 RMSE")
print(f"  DGAT++:    7.90% MAPE / 122.1 RMSE")
print(f"  Δ:        −1.07% MAPE / −14.5 cycles RMSE (12% relative improvement)")

# Wilcoxon signed-rank (non-parametric, more robust for n=6)
try:
    w, pw = stats.wilcoxon(dp, d)
    print(f"\nWilcoxon signed-rank: W={w:.3f}, p={pw:.4f}")
except Exception as e:
    print(f"Wilcoxon error: {e}")
