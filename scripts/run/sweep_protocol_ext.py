"""
Protocol extrapolation experiment: randomly hold out 10/25/50% of MIT charging
protocols (and all the cells that used them) for test. Evaluates whether the
GNN module enables genuine protocol generalisation.

Outputs results JSON per (model, hold_out_frac).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from battery_paper.data import BSEBenchLoader, load_mit_meta
from battery_paper.train import TrainConfig, train, _run_one_split  # noqa
from battery_paper.utils import get_logger

LOG = get_logger("sweep_proto")


def _build_split_proto(meta: dict, all_cells, frac_holdout: float, seed: int):
    """Split by PROTOCOL: hold out `frac_holdout` of unique policies → all cells using
    them go to test."""
    policies = {}
    for c in all_cells:
        if c in meta:
            policies.setdefault(meta[c]["policy"], []).append(c)
    policy_list = sorted(policies.keys())
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(policy_list))
    n_test = max(1, int(round(len(policy_list) * frac_holdout)))
    test_policies = {policy_list[i] for i in perm[:n_test]}
    train_cells, test_cells = [], []
    for p, cs in policies.items():
        (test_cells if p in test_policies else train_cells).extend(cs)
    return train_cells, test_cells, test_policies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--fractions", default="0.10,0.25,0.50")
    ap.add_argument("--models", default="vanilla,full")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--data_root", default="data/raw/mit_hf")
    ap.add_argument("--meta_csv", default="data/interim/mit_meta.csv")
    ap.add_argument("--cache_dir", default="data/processed/mit_cache_full")
    ap.add_argument("--manifest_csv", default="data/interim/mit_manifest_full.csv")
    ap.add_argument("--out_root", default="experiments/sweep_proto_ext")
    args = ap.parse_args()

    fractions = [float(x) for x in args.fractions.split(",")]
    models = args.models.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    loader = BSEBenchLoader(args.data_root)
    meta = load_mit_meta(args.meta_csv)
    all_cells = loader.list_cells()

    summary = []
    for m in models:
        for frac in fractions:
            for s in seeds:
                tag = f"{m}__frac{int(frac*100)}__seed{s}"
                tr, te, test_pol = _build_split_proto(meta, all_cells, frac, s)
                LOG.info("[%s] train=%d cells, test=%d cells, test_policies=%d",
                         tag, len(tr), len(te), len(test_pol))
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
                cfg.hsmm_K = 4
                cfg.hsmm_D_max = 200
                cfg.eval_mode = "5fold"  # we override splits below
                cfg.n_folds = 1
                if m == "vanilla":
                    cfg.model = "vanilla"
                elif m == "full":
                    cfg.model = "hsmm_graph_game"
                    cfg.use_graph = True
                    cfg.alpha_hsmm = 0.001
                    cfg.alpha_aux_proto = 0.1
                else:
                    raise ValueError(m)
                cfg.out_dir = str(out_root / tag)
                Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

                # Direct call to _run_one_split bypassing full train()
                # We need to build the same common_kwargs
                common_kwargs = dict(n_cycles=cfg.n_cycles, intra_len=cfg.intra_len,
                                     features=("voltage_v", "current_a",
                                               "temperature_c", "capacity_ah"),
                                     cache_dir=cfg.cache_dir,
                                     external_meta=meta)
                split_info = {"train": tr, "test1": te, "test2": [], "fold": 0}
                t0 = time.time()
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                try:
                    res = _run_one_split(cfg, loader, common_kwargs, split_info,
                                         Path(cfg.out_dir), device)
                    elapsed = time.time() - t0
                    summary.append({
                        "tag": tag, "model": m, "frac_holdout": frac, "seed": s,
                        "train_size": len(tr), "test_size": len(te),
                        "n_test_policies": len(test_pol),
                        "best_MAPE": res["best"]["MAPE"],
                        "best_RMSE": res["best"]["RMSE"],
                        "best_epoch": res["best"]["epoch"],
                        "final_test_MAPE": res["test1"]["MAPE"],
                        "final_test_RMSE": res["test1"]["RMSE"],
                        "elapsed_s": elapsed,
                    })
                except Exception as e:
                    LOG.exception("%s failed: %s", tag, e)
                    summary.append({"tag": tag, "error": str(e)})
                with open(out_root / "summary.json", "w") as f:
                    json.dump(summary, f, indent=2, default=str)
                LOG.info("[done] %s", tag)

    # Aggregate by (model, frac)
    pd.DataFrame(summary).to_csv(out_root / "summary.csv", index=False)
    LOG.info("ALL DONE: %s", out_root)


if __name__ == "__main__":
    main()
