"""
Extract per-cycle SUMMARY arrays (Qd, Qc, IR, Tavg, Tmin, Tmax, chargetime)
and a fixed-grid ΔQ_{100-10}(V) curve per cell from the Tier-1 raw .mat files.

These are the actual physical fields used in the Severson 2019 paper.
Output: parquet `mit_summary.parquet` (one row per cell, columns include lists).

Output layout:
    cell_id, batch, channel_id, cycle_life, policy_readable,
    cycle (list[int]), Qd (list[float]), Qc (list[float]), IR (list[float]),
    Tavg, Tmax, Tmin, chargetime (each list[float] aligned with cycle),
    dQ100_10 (list[float] length n_v_grid)  -- the discharge ΔQ(V) curve for
                                                 cycle100 - cycle10, on a fixed
                                                 1000-point voltage grid

This package does NOT require torch.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import pandas as pd

from ..utils.logging_utils import get_logger

LOG = get_logger("mit_summary")

BATCH_FILES = {
    "b1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "b2": "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
    "b3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}

# Severson canonical voltage grid: 1000 points across discharge window.
# Their published file ships Qdlin (Q at fixed V) already.
N_V_GRID = 1000


def _to_str(f, ref) -> str:
    arr = np.asarray(f[ref]).flatten()
    return "".join(chr(int(c)) for c in arr if 32 <= int(c) <= 126)


def _to_scalar(f, ref) -> float:
    arr = np.asarray(f[ref]).flatten()
    return float(arr[0]) if len(arr) else float("nan")


def _to_array(f, ref) -> np.ndarray:
    return np.asarray(f[ref]).squeeze()


def _read_summary_arr(f, summ, key) -> np.ndarray:
    """Summary fields in MIT mat are direct datasets, not refs."""
    return np.asarray(summ[key]).squeeze()


def _read_cycle_arr(f, cy, key, k) -> np.ndarray:
    """cycles[k] in MIT mat is a (n_cycles, 1) array of refs."""
    return np.asarray(f[cy[key][k, 0]]).squeeze()


def _iter_cells(mat_path: str, batch_key: str) -> Iterator[dict]:
    with h5py.File(mat_path, "r") as f:
        batch = f["batch"]
        n_cells = batch["cycle_life"].shape[0]
        for i in range(n_cells):
            try:
                cl_arr = np.asarray(f[batch["cycle_life"][i, 0]]).flatten()
                if not len(cl_arr) or np.isnan(cl_arr[0]):
                    continue
                cl = int(cl_arr[0])
                channel = int(np.asarray(f[batch["channel_id"][i, 0]]).flatten()[0])
                policy = _to_str(f, batch["policy_readable"][i, 0])
                # Summary
                summ = f[batch["summary"][i, 0]]
                cycle = _read_summary_arr(f, summ, "cycle").astype(int)
                Qd = _read_summary_arr(f, summ, "QDischarge")
                Qc = _read_summary_arr(f, summ, "QCharge")
                IR = _read_summary_arr(f, summ, "IR")
                Tavg = _read_summary_arr(f, summ, "Tavg")
                Tmax = _read_summary_arr(f, summ, "Tmax")
                Tmin = _read_summary_arr(f, summ, "Tmin")
                chargetime = _read_summary_arr(f, summ, "chargetime")
                # Per-cycle Qdlin
                cy = f[batch["cycles"][i, 0]]
                if "Qdlin" in cy:
                    try:
                        q10 = _read_cycle_arr(f, cy, "Qdlin", 10)
                        q100 = _read_cycle_arr(f, cy, "Qdlin", 100)
                        dq = (q100 - q10).astype(np.float32)
                        if dq.size != N_V_GRID:
                            from scipy.interpolate import interp1d
                            old_x = np.linspace(0, 1, dq.size)
                            new_x = np.linspace(0, 1, N_V_GRID)
                            dq = interp1d(old_x, dq, bounds_error=False,
                                          fill_value=np.nan)(new_x).astype(np.float32)
                    except Exception as e:
                        LOG.warning("Qdlin missing for %s/%d: %s", batch_key, i, e)
                        dq = np.full(N_V_GRID, np.nan, dtype=np.float32)
                else:
                    dq = np.full(N_V_GRID, np.nan, dtype=np.float32)
                yield dict(
                    cell_id=f"{batch_key}c{i}",
                    batch=batch_key,
                    channel_id=channel,
                    cycle_life=cl,
                    policy_readable=policy,
                    n_cycles=int(len(cycle)),
                    cycle=cycle.tolist(),
                    Qd=Qd.astype(float).tolist(),
                    Qc=Qc.astype(float).tolist(),
                    IR=IR.astype(float).tolist(),
                    Tavg=Tavg.astype(float).tolist(),
                    Tmax=Tmax.astype(float).tolist(),
                    Tmin=Tmin.astype(float).tolist(),
                    chargetime=chargetime.astype(float).tolist(),
                    dQ100_10=dq.tolist(),
                )
            except Exception as e:
                LOG.exception("failed cell %s/%d: %s", batch_key, i, e)


def extract_all(raw_dir: str, out_parquet: str) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    rows = []
    for bkey, fname in BATCH_FILES.items():
        p = raw_dir / fname
        if not p.exists():
            LOG.warning("missing %s, skip", p)
            continue
        LOG.info("reading %s", p)
        rows.extend(list(_iter_cells(str(p), bkey)))
    df = pd.DataFrame(rows)
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    LOG.info("wrote %s (%d cells)", out_parquet, len(df))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/mit_raw")
    ap.add_argument("--out", default="data/interim/mit_summary.parquet")
    args = ap.parse_args()
    df = extract_all(args.raw, args.out)
    print(df.head())
    print(df.shape)
    print("first cell Qd len:", len(df["Qd"].iloc[0]),
          "dQ100_10 len:", len(df["dQ100_10"].iloc[0]),
          "has NaN dQ:", any(np.isnan(df["dQ100_10"].iloc[0])))
