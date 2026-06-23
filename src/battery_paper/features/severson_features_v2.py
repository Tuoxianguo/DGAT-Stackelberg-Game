"""
Severson features v2 — built from the official .mat summary fields
(Qd, Qc, IR, Tavg, chargetime) and the per-cell ΔQ_{100-10}(V) curve,
which are EXACTLY the inputs Severson 2019 used. This should reproduce
the published ≈7-12% MAPE numbers.

Usage:
    df_feats = compute_severson_features_v2(df_summary)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_polyfit(x, y, deg=1):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < deg + 1:
        return np.full(deg + 1, np.nan)
    return np.polyfit(x[mask], y[mask], deg)


def compute_severson_features_v2(df: pd.DataFrame, cycle_a: int = 10,
                                 cycle_b: int = 100) -> pd.DataFrame:
    """Compute 9 Severson features from per-cell summary parquet.

    Expected df columns: cell_id, batch, cycle_life, policy_readable,
                         cycle, Qd, Qc, IR, Tavg, chargetime, dQ100_10

    Returns dataframe with one row per cell and feature columns.
    """
    rows = []
    for _, r in df.iterrows():
        cycle = np.asarray(r["cycle"])
        Qd = np.asarray(r["Qd"])
        Qc = np.asarray(r["Qc"])
        IR = np.asarray(r["IR"])
        Tavg = np.asarray(r["Tavg"])
        chargetime = np.asarray(r["chargetime"])
        dQ = np.asarray(r["dQ100_10"])
        feat = {"cell_id": r["cell_id"], "cycle_life": r["cycle_life"],
                "policy": r["policy_readable"], "batch": r["batch"]}

        # --- ΔQ_{100-10}(V) statistics ---
        finite = np.isfinite(dQ)
        if finite.sum() >= 50:
            x = dQ[finite]
            var = float(np.var(x))
            mn = float(np.min(x))
            mean = float(np.mean(x))
            sd = x.std(ddof=0)
            if sd > 0:
                z = (x - mean) / sd
                skew = float((z ** 3).mean())
                kurt = float((z ** 4).mean() - 3)
            else:
                skew = kurt = 0.0
            feat["log_var_dq"] = float(np.log10(max(var, 1e-30)))
            feat["log_abs_min_dq"] = float(np.log10(abs(mn) if mn != 0 else 1e-30))
            feat["skew_dq"] = skew
            feat["kurt_dq"] = kurt
            feat["log_mean_dq"] = float(np.log10(abs(mean) if mean != 0 else 1e-30))
        else:
            for k in ("log_var_dq", "log_abs_min_dq", "skew_dq",
                      "kurt_dq", "log_mean_dq"):
                feat[k] = np.nan

        # --- Capacity fade slope cycle 2-100 ---
        if len(cycle) > cycle_b:
            sub_idx = (cycle >= 2) & (cycle <= cycle_b)
            slope, intercept = _safe_polyfit(cycle[sub_idx].astype(float),
                                             Qd[sub_idx], 1)
            feat["slope_2_100"] = float(slope)
            feat["intercept_2_100"] = float(intercept)
            feat["qd_2"] = float(Qd[1]) if len(Qd) > 1 else np.nan
            feat["qd_100"] = float(Qd[99]) if len(Qd) > 99 else np.nan
            feat["qd_diff_100_2"] = feat["qd_100"] - feat["qd_2"]
        else:
            feat["slope_2_100"] = feat["intercept_2_100"] = feat["qd_2"] = \
                feat["qd_100"] = feat["qd_diff_100_2"] = np.nan

        # --- IR / temperature / charge time ---
        if len(IR) >= cycle_b:
            feat["IR_min_2_100"] = float(np.min(IR[1:cycle_b]))
            feat["IR_at_100_minus_2"] = float(IR[99] - IR[1])
        else:
            feat["IR_min_2_100"] = feat["IR_at_100_minus_2"] = np.nan
        if len(Tavg) >= cycle_b:
            feat["Tavg_integral_2_100"] = float(np.trapz(Tavg[1:cycle_b]))
            feat["Tavg_max_2_100"] = float(np.max(Tavg[1:cycle_b]))
        else:
            feat["Tavg_integral_2_100"] = feat["Tavg_max_2_100"] = np.nan
        if len(chargetime) >= 6:
            feat["chgt_first5"] = float(np.mean(chargetime[1:6]))
        else:
            feat["chgt_first5"] = np.nan
        rows.append(feat)
    return pd.DataFrame(rows)
