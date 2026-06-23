"""
Run the Severson Elastic Net baseline end-to-end on BSEBench parquet data.

Inputs:
  --data_root: directory of *.parquet files (Tier 2 BSEBench)
  --meta_csv:  optional metadata CSV (with cycle_life column).
               If missing, cycle_life is approximated as max(cycle_number).

Outputs:
  data/interim/severson_features.csv
  experiments/run_severson_baseline/{cv_metrics.json, predictions.csv}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from battery_paper.data import BSEBenchLoader, load_mit_meta
from battery_paper.features import build_features_from_summary
from battery_paper.models.baselines import train_severson_elastic_net
from battery_paper.utils import get_logger

LOG = get_logger("run_severson")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/raw/mit_hf")
    ap.add_argument("--meta_csv", default="data/interim/mit_meta.csv")
    ap.add_argument("--feature_csv",
                    default="data/interim/severson_features.csv")
    ap.add_argument("--out_dir",
                    default="experiments/run_severson_baseline")
    ap.add_argument("--cycle_a", type=int, default=10)
    ap.add_argument("--cycle_b", type=int, default=100)
    ap.add_argument("--n_cells_max", type=int, default=None,
                    help="limit cells for quick smoke run (None = all)")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    loader = BSEBenchLoader(args.data_root)
    cells = loader.list_cells()
    if args.n_cells_max:
        cells = cells[:args.n_cells_max]
    LOG.info("using %d cells", len(cells))

    external_meta = {}
    if os.path.exists(args.meta_csv):
        external_meta = load_mit_meta(args.meta_csv)
        LOG.info("external meta available for %d cells", len(external_meta))
    else:
        LOG.warning("meta_csv not found, approximating cycle_life=max(cycle_number)")

    feats = []
    for cid in cells:
        try:
            cell = loader.load_cell(cid, external_meta=external_meta)
            summ = loader.summarize_cell(cid, external_meta=external_meta)
            # If we don't have a Qd column then summarize falls back to size aggregations;
            # detect that case and skip
            if not (summ["Qd_max"] != 0).any():
                LOG.warning("cell %s: no Qd_max info; skipping", cid)
                continue
            f = build_features_from_summary(
                cell, summ, cycle_a=args.cycle_a, cycle_b=args.cycle_b,
                col_v="voltage_v", col_q="capacity_ah", col_i="current_a",
                col_t="temperature_c",
            )
            feats.append(f.to_row())
        except Exception as e:
            LOG.exception("cell %s failed: %s", cid, e)

    df = pd.DataFrame(feats)
    Path(args.feature_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.feature_csv, index=False)
    LOG.info("features written: %s (%d rows)", args.feature_csv, len(df))
    print(df.head())
    print(df.describe())

    # train
    bundle = train_severson_elastic_net(df)
    out = {
        "n_cells_used": int(len(df)),
        "metrics": bundle.cv_metrics,
        "features": bundle.feature_names,
    }
    with open(Path(args.out_dir) / "cv_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    LOG.info("saved metrics -> %s", Path(args.out_dir) / "cv_metrics.json")

    # quick prediction dump (refit and predict on the same set for sanity)
    preds = bundle.predict(df)
    pdf = df[["cell_id", "cycle_life"]].copy()
    pdf["pred_cycle_life"] = preds
    pdf.to_csv(Path(args.out_dir) / "predictions.csv", index=False)
    LOG.info("done")


if __name__ == "__main__":
    main()
