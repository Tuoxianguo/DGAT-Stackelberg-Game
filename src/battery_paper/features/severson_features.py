"""
Severson et al. (Nature Energy 2019) feature set for early life prediction.

Three feature schemas (with diminishing dimensionality):

  "discharge_model"  : 6 features mostly from variance of dQ_100-dQ_10 curve.
  "full_model"       : 9 features adding initial conditions.
  "variance_model"   : 1 feature (log variance of dQ_100-dQ_10).

We compute the *exact* Severson features from per-cycle (V, Qd) interpolated
on a fixed voltage grid. If you have BSEBench parquet, use
    build_features_from_summary(bse_cell, ...)
which auto-handles the alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def _discharge_q_on_v_grid(
    cycle_df: pd.DataFrame,
    v_lo: float = 2.0,
    v_hi: float = 3.5,
    n_pts: int = 1000,
    col_v: str = "voltage_v",
    col_q: str = "discharge_capacity_ah",
    col_i: str = "current_a",
) -> np.ndarray | None:
    """For a single cycle DataFrame, return Q(V) interpolated on common grid (length n_pts).

    We isolate the discharge portion (I < -epsilon), use monotonic V→Q.
    Returns None if the cycle is too short or non-monotonic.
    """
    if col_v not in cycle_df.columns or col_q not in cycle_df.columns:
        return None
    df = cycle_df.copy()
    if col_i in df.columns:
        df = df[df[col_i] < -1e-3]
    if len(df) < 20:
        return None
    df = df.sort_values(col_v, kind="stable")
    df = df.drop_duplicates(subset=[col_v], keep="last")
    v = df[col_v].values
    q = df[col_q].values
    if v[0] > v_lo or v[-1] < v_hi:
        # extrapolate cautiously: clip range to overlapping window
        v_lo_c = max(v_lo, v[0])
        v_hi_c = min(v_hi, v[-1])
        if v_hi_c - v_lo_c < 0.5:
            return None
        v_grid = np.linspace(v_lo_c, v_hi_c, n_pts)
    else:
        v_grid = np.linspace(v_lo, v_hi, n_pts)
    try:
        f = interp1d(v, q, bounds_error=False, fill_value=np.nan)
        return f(v_grid)
    except Exception:
        return None


def compute_dq_variance(q100_vs_v: np.ndarray, q10_vs_v: np.ndarray) -> dict:
    """Severson's primary feature group from ΔQ_{100−10}(V)."""
    dq = q100_vs_v - q10_vs_v
    mask = np.isfinite(dq)
    if mask.sum() < 50:
        return {k: np.nan for k in
                ("var", "min", "skew", "kurt", "mean", "log_var", "abs_min", "log_abs_min")}
    x = dq[mask]
    var = float(np.var(x))
    mn = float(np.min(x))
    mean = float(np.mean(x))
    # skewness / kurtosis (excess), avoid scipy dependency
    sd = x.std(ddof=0)
    if sd > 0:
        z = (x - mean) / sd
        skew = float((z ** 3).mean())
        kurt = float((z ** 4).mean() - 3.0)
    else:
        skew = kurt = 0.0
    abs_min = abs(mn) if mn != 0 else 1e-12
    return dict(
        var=var, log_var=float(np.log10(max(var, 1e-30))),
        min=mn, abs_min=abs_min, log_abs_min=float(np.log10(abs_min)),
        skew=skew, kurt=kurt, mean=mean,
    )


@dataclass
class SeversonFeatures:
    cell_id: str
    feat: dict
    cycle_life: int

    def to_row(self) -> dict:
        out = {"cell_id": self.cell_id, "cycle_life": self.cycle_life}
        out.update(self.feat)
        return out


def build_features_from_summary(
    bse_cell,
    summary_df: pd.DataFrame,
    cycle_a: int = 10,
    cycle_b: int = 100,
    col_v: str = "voltage_v",
    col_q: str = "discharge_capacity_ah",
    col_i: str = "current_a",
    col_t: str = "temperature_c",
) -> SeversonFeatures:
    """Construct Severson "full_model" features (8-9 dim) for one cell.

    Inputs:
      bse_cell      : BSECell from bsebench_loader.load_cell()
      summary_df    : the per-cycle summary (from BSEBenchLoader.summarize_cell)
      cycle_a/b     : Severson uses 10 and 100
    Output: SeversonFeatures
    """
    df_all = bse_cell.df
    for cand in ["cycle_number", "cycle_index", "cycle"]:
        if cand in df_all.columns:
            if cand != "cycle_index":
                df_all = df_all.rename(columns={cand: "cycle_index"})
            break
    else:
        raise KeyError(f"cell {bse_cell.cell_id}: no cycle column")
    a = df_all[df_all["cycle_index"] == cycle_a]
    b = df_all[df_all["cycle_index"] == cycle_b]
    if len(a) == 0 or len(b) == 0:
        return SeversonFeatures(bse_cell.cell_id, {}, bse_cell.cycle_life or -1)
    q_a = _discharge_q_on_v_grid(a, col_v=col_v, col_q=col_q, col_i=col_i)
    q_b = _discharge_q_on_v_grid(b, col_v=col_v, col_q=col_q, col_i=col_i)
    if q_a is None or q_b is None:
        return SeversonFeatures(bse_cell.cell_id, {}, bse_cell.cycle_life or -1)
    vstats = compute_dq_variance(q_b, q_a)

    # Capacity-fade slope between cycle 2 and 100 (Severson "slope_2-100")
    sub = summary_df[(summary_df["cycle"] >= 2) & (summary_df["cycle"] <= cycle_b)]
    if len(sub) > 5 and "Qd_max" in sub.columns:
        x = sub["cycle"].values.astype(float)
        y = sub["Qd_max"].values.astype(float)
        slope, intercept = np.polyfit(x, y, 1)
        qd_2 = float(sub["Qd_max"].iloc[0])
        qd_100 = float(sub["Qd_max"].iloc[-1])
    else:
        slope = intercept = qd_2 = qd_100 = np.nan
    # Initial discharge capacity
    cy2 = summary_df[summary_df["cycle"] == 2]
    qd_init = float(cy2["Qd_max"].iloc[0]) if len(cy2) and "Qd_max" in cy2 else np.nan
    # max-min QD in early cycles
    if "Qd_max" in summary_df.columns:
        early = summary_df[summary_df["cycle"] <= cycle_b]
        qd_diff = float(early["Qd_max"].max() - early["Qd_max"].min())
    else:
        qd_diff = np.nan
    # internal resistance early
    if "Tavg" in summary_df.columns:
        t_avg_2_100 = float(summary_df[(summary_df["cycle"] >= 2) &
                                       (summary_df["cycle"] <= cycle_b)]["Tavg"].mean())
    else:
        t_avg_2_100 = np.nan
    # charge_time (mean over 2-100, Severson uses average of first 5)
    if "charge_time" in summary_df.columns:
        first5 = summary_df[(summary_df["cycle"] >= 2) & (summary_df["cycle"] <= 6)]
        chgt5 = float(first5["charge_time"].mean()) if len(first5) else np.nan
    else:
        chgt5 = np.nan

    feat = dict(
        log_var_dq=vstats["log_var"],
        log_abs_min_dq=vstats["log_abs_min"],
        skew_dq=vstats["skew"],
        kurt_dq=vstats["kurt"],
        slope_2_100=float(slope),
        intercept_2_100=float(intercept),
        qd_2=float(qd_2),
        qd_100=float(qd_100),
        qd_init=float(qd_init),
        qd_max_min_early=float(qd_diff),
        t_avg_2_100=float(t_avg_2_100),
        chgt_first5=float(chgt5),
    )
    return SeversonFeatures(bse_cell.cell_id, feat, bse_cell.cycle_life or -1)
