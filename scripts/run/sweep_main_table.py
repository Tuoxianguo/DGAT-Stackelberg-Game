"""
Run the full main-table experiment matrix in one go (on a single T4 GPU).

Matrix:
  - models = {vanilla, hsmm_only, graph_only, hsmm_graph, full}
  - splits = {severson_batch, 5fold}

Each config trains for `epochs` epochs and writes results to
`out_root/<model>__<split>/`.

This script is meant to be invoked once via:
    python scripts/run/sweep_main_table.py --epochs 60
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from battery_paper.train import TrainConfig, train
from battery_paper.utils import get_logger

LOG = get_logger("sweep")


CONFIGS = {
    "vanilla": dict(model="vanilla"),
    "hsmm_only": dict(model="hsmm_graph_game", use_graph=False,
                      alpha_hsmm=0.001, alpha_aux_proto=0.0),
    "graph_only": dict(model="hsmm_graph_game", use_graph=True,
                       alpha_hsmm=0.0, alpha_aux_proto=0.05),
    "hsmm_graph": dict(model="hsmm_graph_game", use_graph=True,
                       alpha_hsmm=0.001, alpha_aux_proto=0.05),
    "full":      dict(model="hsmm_graph_game", use_graph=True,
                      alpha_hsmm=0.001, alpha_aux_proto=0.1),
    # NEW HYBRID configs (with Severson aux features)
    "vanilla_hyb": dict(model="vanilla",
                        hybrid_features_csv="experiments/run_severson_v2/features_v2.csv",
                        loss_type="tail_weighted_huber", tail_weight_power=1.0),
    "graph_hyb":   dict(model="hsmm_graph_game", use_graph=True,
                        alpha_hsmm=0.0, alpha_aux_proto=0.02,
                        hybrid_features_csv="experiments/run_severson_v2/features_v2.csv",
                        loss_type="tail_weighted_huber", tail_weight_power=1.0),
    "full_hyb":    dict(model="hsmm_graph_game", use_graph=True,
                        alpha_hsmm=0.0005, alpha_aux_proto=0.05,
                        hybrid_features_csv="experiments/run_severson_v2/features_v2.csv",
                        loss_type="tail_weighted_huber", tail_weight_power=1.0),
}

SPLITS = {
    "severson": dict(eval_mode="severson"),
    "5fold":    dict(eval_mode="5fold", n_folds=5),
    "stratified": dict(eval_mode="stratified_5fold", n_folds=5),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out_root", default="experiments/sweep_main")
    ap.add_argument("--data_root", default="data/raw/mit_hf")
    ap.add_argument("--meta_csv", default="data/interim/mit_meta.csv")
    ap.add_argument("--cache_dir", default="data/processed/mit_cache_full")
    ap.add_argument("--manifest_csv", default="data/interim/mit_manifest_full.csv")
    ap.add_argument("--only_models", default=None, help="comma-separated")
    ap.add_argument("--only_splits", default=None)
    args = ap.parse_args()

    models = (args.only_models.split(",") if args.only_models
              else list(CONFIGS.keys()))
    splits = (args.only_splits.split(",") if args.only_splits
              else list(SPLITS.keys()))

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split_name in splits:
        for model_name in models:
            tag = f"{model_name}__{split_name}"
            cfg = TrainConfig()
            cfg.data_root = args.data_root
            cfg.cache_dir = args.cache_dir
            cfg.meta_csv = args.meta_csv
            cfg.manifest_csv = args.manifest_csv
            cfg.epochs = args.epochs
            cfg.batch_size = 8
            cfg.lr = 3e-4
            cfg.weight_decay = 1e-3
            cfg.augment = True
            cfg.n_cycles = 100
            cfg.intra_len = 64
            cfg.d_model = 96
            cfg.n_layers = 3
            cfg.n_heads = 4
            cfg.hsmm_K = 4
            cfg.hsmm_D_max = 200
            for k, v in CONFIGS[model_name].items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
                else:
                    # also support attributes not yet in TrainConfig (defensively)
                    setattr(cfg, k, v)
            for k, v in SPLITS[split_name].items():
                setattr(cfg, k, v)
            cfg.out_dir = str(out_root / tag)
            t0 = time.time()
            LOG.info("\n%s\n=== RUNNING  %s  ===", "=" * 60, tag)
            try:
                res = train(cfg)
                summary[tag] = {
                    "elapsed_s": time.time() - t0,
                    "result": res if isinstance(res, dict) else str(res),
                }
            except Exception as e:
                LOG.exception("run %s failed: %s", tag, e)
                summary[tag] = {"error": str(e),
                                 "elapsed_s": time.time() - t0}
            with open(out_root / "sweep_summary.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)
            LOG.info("[done] %s in %.1fs", tag, time.time() - t0)

    LOG.info("\nALL DONE. Summary:")
    for k, v in summary.items():
        if "result" in v and isinstance(v["result"], dict):
            r = v["result"]
            if "best" in r:
                LOG.info("  %-35s  best ep %3d  MAPE=%.2f%%",
                         k, r["best"]["epoch"], r["best"]["MAPE"])
            elif "MAPE_test1_mean" in r:
                LOG.info("  %-35s  5fold MAPE=%.2f ± %.2f%%",
                         k, r["MAPE_test1_mean"], r["MAPE_test1_std"])
            else:
                LOG.info("  %-35s  result keys: %s", k, list(r.keys()))
        else:
            LOG.info("  %-35s  ERROR / partial", k)


if __name__ == "__main__":
    main()
