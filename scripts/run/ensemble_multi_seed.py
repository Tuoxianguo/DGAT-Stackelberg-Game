"""
Ensemble predictions across multiple random seeds for the same fold splits.

We use 4 seeds: 42 (from sweep_v5_random) + 7, 2026, 1024 (from multi_seed_graph).
Per fold, we predict with all 4 ckpts and average the predictions (log-space mean
for log-normal targets gives geometric mean of preds, which is more robust than
arithmetic).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from battery_paper.data import BSEBenchLoader, BSEEarlyPredictDataset, load_mit_meta
from battery_paper.models.baselines import VanillaTransformerRUL
from battery_paper.models.proposed import HSMMGraphGameModel


def _load_model(ckpt_path: Path, device="cuda"):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state["cfg"]
    F_in = len(cfg["features"])
    d_aux = 0
    if cfg["model"] == "vanilla":
        model = VanillaTransformerRUL(in_features=F_in, intra_len=cfg["intra_len"],
                                      d_model=cfg["d_model"], n_layers=cfg["n_layers"],
                                      n_heads=cfg["n_heads"],
                                      d_aux_feat=d_aux).to(device)
    elif cfg["model"] == "lstm":
        from battery_paper.models.baselines import LSTMRUL
        model = LSTMRUL(in_features=F_in, intra_len=cfg["intra_len"],
                        d_model=cfg["d_model"], n_layers=cfg["n_layers"],
                        d_aux_feat=d_aux).to(device)
    elif cfg["model"] == "lstm_att":
        from battery_paper.models.baselines import LSTMAttRUL
        model = LSTMAttRUL(in_features=F_in, intra_len=cfg["intra_len"],
                           d_model=cfg["d_model"], n_layers=cfg["n_layers"],
                           n_heads=cfg["n_heads"], d_aux_feat=d_aux).to(device)
    elif cfg["model"] == "battery_gpt":
        from battery_paper.models.baselines import BatteryGPTLite
        model = BatteryGPTLite(in_features=F_in, intra_len=cfg["intra_len"],
                               d_model=cfg["d_model"], n_layers=cfg["n_layers"],
                               n_heads=cfg["n_heads"], d_aux_feat=d_aux).to(device)
    elif cfg["model"] == "pbt":
        from battery_paper.models.baselines import PBTLite
        model = PBTLite(in_features=F_in, intra_len=cfg["intra_len"],
                        d_model=cfg["d_model"], n_layers=cfg["n_layers"],
                        n_heads=cfg["n_heads"], d_aux_feat=d_aux).to(device)
    elif cfg["model"] == "dgat":
        from battery_paper.models.baselines import DGATLite
        model = DGATLite(in_features=F_in, intra_len=cfg["intra_len"],
                         d_model=cfg["d_model"], n_layers=cfg["n_layers"],
                         n_heads=cfg["n_heads"], d_aux_feat=d_aux).to(device)
    else:
        model = HSMMGraphGameModel(
            in_features=F_in, intra_len=cfg["intra_len"],
            d_model=cfg["d_model"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
            hsmm_K=cfg["hsmm_K"], hsmm_D_max=cfg["hsmm_D_max"],
            use_graph=cfg["use_graph"], d_aux_feat=d_aux,
        ).to(device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    return model, cfg


def _predict(model, x, m, p, fa=None):
    from battery_paper.models.baselines import (LSTMRUL, LSTMAttRUL,
                                                BatteryGPTLite, PBTLite, DGATLite)
    _SIMPLE = (VanillaTransformerRUL, LSTMRUL, LSTMAttRUL,
               BatteryGPTLite, PBTLite, DGATLite)
    with torch.no_grad():
        if isinstance(model, _SIMPLE):
            return model(x, m, fa) if fa is not None else model(x, m)
        else:
            out = model(x, m, p, fa) if fa is not None else model(x, m, p)
            return out.rul_hat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/raw/mit_hf")
    ap.add_argument("--meta_csv", default="data/interim/mit_meta.csv")
    ap.add_argument("--cache_dir", default="data/processed/mit_cache_full")
    ap.add_argument("--out", default="experiments/multi_seed_ensemble.json")
    args = ap.parse_args()

    # checkpoint paths: one per (seed, fold)
    import os
    model_tag = os.environ.get("MODEL_TAG", "graph_only")
    root = os.environ.get("MS_ROOT", "experiments/v6_final")
    seeds_env = os.environ.get("SEEDS", "42,7,2026,1024")
    seed_paths = {int(s): f"{root}/{model_tag}__seed{s}"
                  for s in seeds_env.split(",")}

    # Build fold cell ids (must match sweep splits - random KFold seed=42 is the same
    # for all multi_seed runs because they all use eval_mode='5fold' with seed=42)
    # Wait - actually multi_seed sweeps used their own seed for KFold permutation.
    # So we need to reconstruct splits per seed.
    from sklearn.model_selection import KFold
    manifest = pd.read_csv("data/interim/mit_manifest_full.csv")
    manifest = manifest.dropna(subset=["cycle_life"])
    manifest = manifest[manifest["cycle_life"] > 100]
    cells_all_unshuffled = manifest["cell_id"].tolist()

    meta = load_mit_meta(args.meta_csv)
    loader = BSEBenchLoader(args.data_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    per_seed_preds = {}  # (seed, cell_id) -> pred
    per_cell_truth = {}

    for seed, run_dir in seed_paths.items():
        run_dir = Path(run_dir)
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(cells_all_unshuffled))
        cells_shuffled = [cells_all_unshuffled[i] for i in perm]
        kf = KFold(n_splits=5)
        for fi, (_, te_idx) in enumerate(kf.split(cells_shuffled)):
            test_cells = [cells_shuffled[i] for i in te_idx]
            ckpt = run_dir / f"fold_{fi}" / "best.pt"
            if not ckpt.exists():
                print(f"  missing {ckpt}")
                continue
            model, cfg = _load_model(ckpt, device=device)
            ds = BSEEarlyPredictDataset(loader, test_cells, n_cycles=100,
                                        intra_len=64,
                                        features=cfg["features"],
                                        cache_dir=args.cache_dir,
                                        external_meta=meta, augment=False)
            xs, ms, ps, ys = [], [], [], []
            for i in range(len(ds)):
                r = ds[i]
                xs.append(r["x"]); ms.append(r["mask"])
                ps.append(r["proto"]); ys.append(r["y"])
            if not xs:
                continue
            x = torch.stack(xs).to(device); mk = torch.stack(ms).to(device)
            p = torch.stack(ps).to(device)
            yhat = _predict(model, x, mk, p).cpu().numpy()
            for j, cid in enumerate(test_cells):
                per_seed_preds[(seed, cid)] = float(yhat[j])
                per_cell_truth[cid] = float(ys[j].item())
            del model
        torch.cuda.empty_cache()

    # Per-seed MAPE (for reference)
    print("\n=== Per-seed MAPE (whole test set) ===")
    for seed in seed_paths:
        preds = []; trues = []
        for cid, y in per_cell_truth.items():
            if (seed, cid) in per_seed_preds:
                preds.append(per_seed_preds[(seed, cid)])
                trues.append(y)
        preds = np.array(preds); trues = np.array(trues)
        if len(preds):
            mape = np.mean(np.abs(preds - trues) / np.clip(trues, 1, None)) * 100
            rmse = np.sqrt(np.mean((preds - trues) ** 2))
            print(f"  seed={seed:5d}  MAPE={mape:.2f}%  RMSE={rmse:.1f}  n={len(preds)}")

    # Ensemble: geometric mean (log-mean) per cell across seeds
    print("\n=== Geometric-mean ENSEMBLE (4 seeds) ===")
    ens_preds = []; ens_trues = []
    per_cell_records = []
    for cid, y in per_cell_truth.items():
        pset = [per_seed_preds[(s, cid)] for s in seed_paths if (s, cid) in per_seed_preds]
        if not pset:
            continue
        # Geometric mean (log mean)
        ge = float(np.exp(np.mean(np.log(np.clip(np.array(pset), 1, None)))))
        ens_preds.append(ge); ens_trues.append(y)
        per_cell_records.append({"cell_id": cid, "y_true": y,
                                 "preds": pset, "ensemble": ge,
                                 "ape": abs(ge - y) / y * 100})
    ens_preds = np.array(ens_preds); ens_trues = np.array(ens_trues)
    mape = float(np.mean(np.abs(ens_preds - ens_trues) / np.clip(ens_trues, 1, None)) * 100)
    rmse = float(np.sqrt(np.mean((ens_preds - ens_trues) ** 2)))
    print(f"  ENSEMBLE  MAPE={mape:.2f}%  RMSE={rmse:.1f}  n={len(ens_preds)}")

    # Trimmed mean (drop best+worst pred per cell)
    print("\n=== Trimmed-mean ENSEMBLE (drop best+worst per cell) ===")
    tr_preds = []; tr_trues = []
    for r in per_cell_records:
        ps = sorted(r["preds"])
        if len(ps) >= 3:
            keep = ps[1:-1]
        else:
            keep = ps
        tm = float(np.exp(np.mean(np.log(np.clip(np.array(keep), 1, None)))))
        tr_preds.append(tm); tr_trues.append(r["y_true"])
    tr_preds = np.array(tr_preds); tr_trues = np.array(tr_trues)
    mape_t = float(np.mean(np.abs(tr_preds - tr_trues) / np.clip(tr_trues, 1, None)) * 100)
    rmse_t = float(np.sqrt(np.mean((tr_preds - tr_trues) ** 2)))
    print(f"  TRIMMED   MAPE={mape_t:.2f}%  RMSE={rmse_t:.1f}")

    # Median
    print("\n=== Median ENSEMBLE ===")
    md_preds = np.array([float(np.median(r["preds"])) for r in per_cell_records])
    md_trues = np.array([r["y_true"] for r in per_cell_records])
    mape_m = float(np.mean(np.abs(md_preds - md_trues) / np.clip(md_trues, 1, None)) * 100)
    rmse_m = float(np.sqrt(np.mean((md_preds - md_trues) ** 2)))
    print(f"  MEDIAN    MAPE={mape_m:.2f}%  RMSE={rmse_m:.1f}")

    out = {
        "ensemble_geometric_mean": {"MAPE": mape, "RMSE": rmse},
        "ensemble_trimmed_mean":   {"MAPE": mape_t, "RMSE": rmse_t},
        "ensemble_median":         {"MAPE": mape_m, "RMSE": rmse_m},
        "n_cells": len(ens_preds),
        "n_seeds": len(seed_paths),
        "per_cell": per_cell_records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
