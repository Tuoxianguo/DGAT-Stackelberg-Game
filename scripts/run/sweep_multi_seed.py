"""
Multi-seed ensemble: run the best single model (graph_only) with N different
random seeds, then ensemble the predictions per-cell.

Predicted gain: ~1% MAPE reduction from variance averaging.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from battery_paper.train import TrainConfig, train
from battery_paper.utils import get_logger

LOG = get_logger("sweep_seed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seeds", default="42,7,2026,123,1024")
    ap.add_argument("--model", default="graph_only",
                    choices=["graph_only", "full", "vanilla", "lstm", "lstm_att",
                             "battery_gpt", "pbt", "dgat", "dgat_plus",
                             "dgat_plus_hsmm", "dgat_plus_graph", "dgat_plus_full"])
    ap.add_argument("--out_root",
                    default="experiments/sweep_multi_seed")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    base_cfg = dict(
        data_root="data/raw/mit_hf",
        cache_dir="data/processed/mit_cache_full",
        meta_csv="data/interim/mit_meta.csv",
        manifest_csv="data/interim/mit_manifest_full.csv",
        n_cycles=100, intra_len=64, d_model=96, n_layers=3, n_heads=4,
        hsmm_K=4, hsmm_D_max=200,
        epochs=args.epochs, batch_size=8, lr=3e-4, weight_decay=1e-3,
        augment=True, eval_mode="5fold", n_folds=5,
    )

    if args.model == "vanilla":
        base_cfg["model"] = "vanilla"
    elif args.model == "lstm":
        base_cfg["model"] = "lstm"
    elif args.model == "lstm_att":
        base_cfg["model"] = "lstm_att"
    elif args.model == "battery_gpt":
        base_cfg["model"] = "battery_gpt"
    elif args.model == "pbt":
        base_cfg["model"] = "pbt"
    elif args.model == "dgat":
        base_cfg["model"] = "dgat"
    elif args.model == "dgat_plus":
        base_cfg["model"] = "dgat_plus"
    elif args.model == "dgat_plus_hsmm":
        base_cfg["model"] = "dgat_plus_hsmm"
        base_cfg["use_graph"] = False
        base_cfg["alpha_hsmm"] = 0.0005
        base_cfg["alpha_aux_proto"] = 0.0
    elif args.model == "dgat_plus_graph":
        base_cfg["model"] = "dgat_plus_graph"
        base_cfg["use_graph"] = True
        base_cfg["alpha_hsmm"] = 0.0
        base_cfg["alpha_aux_proto"] = 0.05
    elif args.model == "dgat_plus_full":
        base_cfg["model"] = "dgat_plus_full"
        base_cfg["use_graph"] = True
        base_cfg["alpha_hsmm"] = 0.0005
        base_cfg["alpha_aux_proto"] = 0.05
    elif args.model == "graph_only":
        base_cfg["model"] = "hsmm_graph_game"
        base_cfg["use_graph"] = True
        base_cfg["alpha_hsmm"] = 0.0
        base_cfg["alpha_aux_proto"] = 0.05
    elif args.model == "full":
        base_cfg["model"] = "hsmm_graph_game"
        base_cfg["use_graph"] = True
        base_cfg["alpha_hsmm"] = 0.001
        base_cfg["alpha_aux_proto"] = 0.1

    summary = {}
    for s in seeds:
        tag = f"{args.model}__seed{s}"
        LOG.info("\n%s\n=== %s ===", "=" * 60, tag)
        cfg = TrainConfig()
        for k, v in base_cfg.items():
            setattr(cfg, k, v)
        cfg.seed = s
        cfg.out_dir = str(out_root / tag)
        t0 = time.time()
        try:
            res = train(cfg)
            summary[tag] = {
                "seed": s,
                "elapsed_s": time.time() - t0,
                "MAPE_mean": res.get("MAPE_test1_mean") if isinstance(res, dict) else None,
                "MAPE_std":  res.get("MAPE_test1_std")  if isinstance(res, dict) else None,
                "RMSE_mean": res.get("RMSE_test1_mean") if isinstance(res, dict) else None,
                "RMSE_std":  res.get("RMSE_test1_std")  if isinstance(res, dict) else None,
            }
        except Exception as e:
            LOG.exception("seed %d failed: %s", s, e)
            summary[tag] = {"error": str(e)}
        with open(out_root / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    LOG.info("\nDONE. Single-seed MAPE per seed:")
    for tag, r in summary.items():
        if r.get("MAPE_mean") is not None:
            LOG.info("  %-25s MAPE %.2f ± %.2f%%",
                     tag, r["MAPE_mean"], r["MAPE_std"])


if __name__ == "__main__":
    main()
