"""
Extract cycle_life + policy_readable for each cell from the Tier-1 raw
.mat files (bsebench-org/severson-2019-raw). Output: a CSV that the
BSEBench parquet loader can JOIN against.

This script does NOT require torch and is much smaller in scope than the
full MITLoader (we only pull summary metadata, skipping per-cycle arrays).
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

LOG = get_logger("mit_meta")

# Tier-1 raw file names → BSEBench batch key
BATCH_FILES = {
    "b1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "b2": "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
    "b3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}


def _to_str(f: h5py.File, ref) -> str:
    arr = np.asarray(f[ref]).flatten()
    return "".join(chr(int(c)) for c in arr if 32 <= int(c) <= 126)


def _to_scalar(f: h5py.File, ref) -> float:
    arr = np.asarray(f[ref]).flatten()
    return float(arr[0]) if len(arr) else float("nan")


def _iter_cells(mat_path: str, batch_key: str) -> Iterator[dict]:
    with h5py.File(mat_path, "r") as f:
        batch = f["batch"]
        n_cells = batch["cycle_life"].shape[0]
        for i in range(n_cells):
            try:
                cl = int(_to_scalar(f, batch["cycle_life"][i, 0]))
                barcode = _to_str(f, batch["barcode"][i, 0])
                channel = int(_to_scalar(f, batch["channel_id"][i, 0]))
                policy = _to_str(f, batch["policy_readable"][i, 0])
                yield dict(
                    cell_id=f"{batch_key}c{i}",
                    batch=batch_key,
                    channel_id=channel,
                    barcode=barcode,
                    cycle_life=cl,
                    policy_readable=policy,
                )
            except Exception as e:  # noqa
                LOG.exception("failed cell %s/%d: %s", batch_key, i, e)


def extract_all(raw_dir: str, out_csv: str) -> pd.DataFrame:
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
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    LOG.info("wrote %s (%d rows)", out_csv, len(df))
    return df


def load_meta(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    out = {}
    for _, r in df.iterrows():
        out[r["cell_id"]] = {
            "cycle_life": int(r["cycle_life"]),
            "policy": r["policy_readable"],
            "channel_id": int(r["channel_id"]),
            "barcode": r["barcode"],
        }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/mit_raw")
    ap.add_argument("--out", default="data/interim/mit_meta.csv")
    args = ap.parse_args()
    df = extract_all(args.raw, args.out)
    print(df.head())
    print(f"\n=== summary ===")
    print(df.groupby("batch").agg(
        n_cells=("cell_id", "count"),
        cycle_life_min=("cycle_life", "min"),
        cycle_life_med=("cycle_life", "median"),
        cycle_life_max=("cycle_life", "max"),
        n_policies=("policy_readable", "nunique"),
    ))
