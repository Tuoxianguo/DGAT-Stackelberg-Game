"""
Ensemble + Test-Time Augmentation on all trained 5-fold checkpoints.

For each fold's test set:
  - Load each ablation model's best.pt checkpoint trained on that fold's train.
  - Predict (mean, std) across multiple TTA passes.
  - Ensemble across {vanilla, hsmm_only, graph_only, hsmm_graph, full} by mean.

Outputs:
  - ensemble_results.json with per-fold and aggregate MAPE/RMSE
  - per-cell predictions CSV
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
    if cfg["model"] == "vanilla":
        model = VanillaTransformerRUL(in_features=F_in, intra_len=cfg["intra_len"],
                                      d_model=cfg["d_model"], n_layers=cfg["n_layers"],
                                      n_heads=cfg["n_heads"]).to(device)
    else:
        model = HSMMGraphGameModel(
            in_features=F_in, intra_len=cfg["intra_len"],
            d_model=cfg["d_model"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
            hsmm_K=cfg["hsmm_K"], hsmm_D_max=cfg["hsmm_D_max"],
            use_graph=cfg["use_graph"],
        ).to(device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    return model, cfg


def _predict_tta(model, x, m, p, n_tta=5, noise_std=0.005):
    """Average predictions over n_tta noisy variants.
    First pass is always noise-free (clean prediction).
    """
    preds = []
    for i in range(n_tta):
        if i == 0:
            xn = x
        else:
            xn = x + torch.randn_like(x) * noise_std * x.std()
        with torch.no_grad():
            if isinstance(model, VanillaTransformerRUL):
                yhat = model(xn, m)
            else:
                out = model(xn, m, p)
                yhat = out.rul_hat
        preds.append(yhat.cpu().numpy())
    return np.stack(preds, 0)  # (n_tta, B)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_root", default="experiments/sweep_v3_5fold")
    ap.add_argument("--data_root", default="data/raw/mit_hf")
    ap.add_argument("--meta_csv", default="data/interim/mit_meta.csv")
    ap.add_argument("--cache_dir", default="data/processed/mit_cache_full")
    ap.add_argument("--n_tta", type=int, default=5)
    ap.add_argument("--out", default="experiments/sweep_v3_5fold/ensemble.json")
    args = ap.parse_args()

    sweep_root = Path(args.sweep_root)
    models = ["vanilla", "hsmm_only", "graph_only", "hsmm_graph", "full"]
    meta = load_mit_meta(args.meta_csv)
    loader = BSEBenchLoader(args.data_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Per-model fold predictions
    per_model_preds = {}
    all_truths = {}
    for m_name in models:
        run_dir = sweep_root / f"{m_name}__5fold"
        if not run_dir.exists():
            continue
        fold_preds = {}
        for fold_dir in sorted(run_dir.glob("fold_*")):
            fold_idx = int(fold_dir.name.split("_")[1])
            ckpt = fold_dir / "best.pt"
            results_json = fold_dir / "results.json"
            if not ckpt.exists() or not results_json.exists():
                print(f"  missing {fold_dir}")
                continue
            results = json.loads(results_json.read_text())
            split_size = results.get("split_size", {})
            # we need to reconstruct the test cell list — we re-derive via the same KFold seed
            # but easier: read the json's history cells? Not there.
            # Workaround: load best ckpt, run it on the manifest cells fold by fold,
            # use same KFold(seed=42) split as in train.py.
            model, cfg = _load_model(ckpt, device=device)
            fold_preds[fold_idx] = (model, cfg)
        per_model_preds[m_name] = fold_preds

    if not per_model_preds:
        print("NO CHECKPOINTS FOUND, exiting")
        return

    from sklearn.model_selection import KFold
    manifest = pd.read_csv("data/interim/mit_manifest_full.csv")
    manifest = manifest.dropna(subset=["cycle_life"])
    manifest = manifest[manifest["cycle_life"] > 100]
    cells_all = manifest["cell_id"].tolist()
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(cells_all))
    cells_all = [cells_all[i] for i in perm]
    splits = list(KFold(n_splits=5).split(cells_all))

    ensemble_per_fold = {}
    per_cell_records = []
    for fold_idx, (tr_idx, te_idx) in enumerate(splits):
        test_cells = [cells_all[i] for i in te_idx]
        ds = BSEEarlyPredictDataset(loader, test_cells, n_cycles=100, intra_len=64,
                                    features=("voltage_v", "current_a",
                                              "temperature_c", "capacity_ah"),
                                    cache_dir=args.cache_dir,
                                    external_meta=meta, augment=False)
        xs, ms, ps, ys = [], [], [], []
        for i in range(len(ds)):
            r = ds[i]
            xs.append(r["x"]); ms.append(r["mask"]); ps.append(r["proto"]); ys.append(r["y"])
        x = torch.stack(xs).to(device)
        mk = torch.stack(ms).to(device)
        p = torch.stack(ps).to(device)
        y = torch.stack(ys).numpy()
        all_truths[fold_idx] = y

        all_model_preds = []
        per_model = {}
        for m_name in models:
            if m_name not in per_model_preds or fold_idx not in per_model_preds[m_name]:
                continue
            model, cfg = per_model_preds[m_name][fold_idx]
            preds = _predict_tta(model, x, mk, p, n_tta=args.n_tta)
            pred_mean = preds.mean(0)
            per_model[m_name] = pred_mean
            all_model_preds.append(pred_mean)
        if not all_model_preds:
            continue
        ensemble = np.mean(np.stack(all_model_preds, 0), axis=0)
        per_model["ensemble"] = ensemble
        ensemble_per_fold[fold_idx] = per_model
        for i, cid in enumerate(test_cells):
            rec = {"cell_id": cid, "y_true": float(y[i])}
            for k, v in per_model.items():
                rec[f"pred_{k}"] = float(v[i])
            per_cell_records.append(rec)

    # Aggregate
    summary = {}
    for m_name in list(models) + ["ensemble"]:
        all_preds = []; all_trues = []
        for fold_idx, preds in ensemble_per_fold.items():
            if m_name not in preds:
                continue
            all_preds.append(preds[m_name])
            all_trues.append(all_truths[fold_idx])
        if not all_preds:
            continue
        ap_ = np.concatenate(all_preds); at_ = np.concatenate(all_trues)
        mape = float(np.mean(np.abs(ap_ - at_) / np.clip(at_, 1, None)) * 100)
        rmse = float(np.sqrt(np.mean((ap_ - at_) ** 2)))
        summary[m_name] = {"MAPE": mape, "RMSE": rmse, "n": len(ap_)}

    print("\n=== Ensemble + TTA summary (5-fold pooled) ===")
    for k, v in summary.items():
        print(f"  {k:15s}  MAPE={v['MAPE']:.2f}%   RMSE={v['RMSE']:.1f}   n={v['n']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"per_model": summary,
               "per_cell": per_cell_records},
              open(args.out, "w"), indent=2)
    print("\nSaved to", args.out)


if __name__ == "__main__":
    main()
