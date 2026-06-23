"""
Cross-domain evaluation: MIT-trained Vanilla CT → HUST (NCA/NCM/NCM_NCA).

Uses 8-seed × 5-fold = 40 Vanilla CT checkpoints from v6_vanilla,
ensembles their predictions on HUST cells (no fine-tuning, zero-shot).

Per chemistry sub-dataset MAPE is reported separately + overall.
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

from battery_paper.data.hust_loader import load_hust_cells
from battery_paper.models.baselines import VanillaTransformerRUL


def _load_vanilla(ckpt_path: Path, device="cuda"):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state["cfg"]
    F_in = len(cfg["features"])
    model = VanillaTransformerRUL(in_features=F_in, intra_len=cfg["intra_len"],
                                  d_model=cfg["d_model"], n_layers=cfg["n_layers"],
                                  n_heads=cfg["n_heads"], d_aux_feat=0).to(device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hust_root", default="data/raw/hust")
    ap.add_argument("--ckpt_root", default="experiments/v6_vanilla")
    ap.add_argument("--seeds", default="42,7,2026,1024,100,777")
    ap.add_argument("--max_per_subset", type=int, default=15)
    ap.add_argument("--n_tta", type=int, default=5)
    ap.add_argument("--noise_factor", type=float, default=0.003)
    ap.add_argument("--out", default="experiments/hust_crossdomain.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    print(f"Loading HUST cells (up to {args.max_per_subset} per subset)...")
    cells = load_hust_cells(args.hust_root, n_cycles=100, intra_len=64,
                            max_cells_per_subset=args.max_per_subset)
    print(f"Loaded {len(cells)} HUST cells\n")
    if not cells:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Pre-stack tensors
    xs = torch.tensor(np.stack([c["x"] for c in cells]), dtype=torch.float32).to(device)
    ms = torch.tensor(np.stack([c["mask"] for c in cells]), dtype=torch.float32).to(device)
    ys = np.array([c["y"] for c in cells])
    chems = [c["chem"] for c in cells]
    cell_ids = [c["cell_id"] for c in cells]

    # For each (seed, fold), load ckpt and predict. Then ensemble.
    per_seed_predictions = {}   # seed -> (n_cells,) array (averaged over folds + TTA)
    ckpt_root = Path(args.ckpt_root)
    for seed in seeds:
        per_fold_preds = []
        for fi in range(5):
            ckpt = ckpt_root / f"vanilla__seed{seed}" / f"fold_{fi}" / "best.pt"
            if not ckpt.exists():
                print(f"  missing {ckpt}")
                continue
            model, cfg = _load_vanilla(ckpt, device=device)
            # TTA: 1 clean + (n_tta-1) noisy passes
            preds_log = []
            with torch.no_grad():
                preds_log.append(np.log(np.clip(model(xs, ms).cpu().numpy(), 1, None)))
                for _ in range(args.n_tta - 1):
                    xn = xs + torch.randn_like(xs) * args.noise_factor * xs.std()
                    preds_log.append(np.log(np.clip(model(xn, ms).cpu().numpy(), 1, None)))
            pred = np.exp(np.mean(np.stack(preds_log, 0), 0))
            per_fold_preds.append(pred)
            del model
        if per_fold_preds:
            # Median across folds
            per_seed_predictions[seed] = np.median(np.stack(per_fold_preds, 0), 0)
        torch.cuda.empty_cache()

    # Ensemble across seeds: median
    all_preds = np.stack(list(per_seed_predictions.values()), 0)  # (n_seeds, n_cells)
    ensemble = np.median(all_preds, 0)

    # Per-chemistry MAPE
    chem_results = {}
    for ch in sorted(set(chems)):
        mask = np.array([c == ch for c in chems])
        if mask.sum() == 0:
            continue
        p = ensemble[mask]; y = ys[mask]
        mape = float(np.mean(np.abs(p - y) / np.clip(y, 1, None)) * 100)
        rmse = float(np.sqrt(np.mean((p - y) ** 2)))
        chem_results[ch] = {"n": int(mask.sum()), "MAPE": mape, "RMSE": rmse,
                            "y_mean": float(y.mean()), "pred_mean": float(p.mean())}

    overall_mape = float(np.mean(np.abs(ensemble - ys) / np.clip(ys, 1, None)) * 100)
    overall_rmse = float(np.sqrt(np.mean((ensemble - ys) ** 2)))
    print(f"\n=== HUST cross-domain (MIT Vanilla CT zero-shot, n={len(cells)}) ===")
    print(f"  Overall    MAPE = {overall_mape:.2f}%, RMSE = {overall_rmse:.1f}")
    for ch, r in chem_results.items():
        print(f"  {ch:10s}  n={r['n']:3d}  MAPE={r['MAPE']:6.2f}%  "
              f"RMSE={r['RMSE']:6.1f}  y_mean={r['y_mean']:6.0f}  "
              f"pred_mean={r['pred_mean']:6.0f}")

    # Per-seed MAPE for comparison
    print(f"\n  per-seed MAPE (for reference):")
    for seed, p in per_seed_predictions.items():
        m = float(np.mean(np.abs(p - ys) / np.clip(ys, 1, None)) * 100)
        print(f"    seed={seed:5d}  MAPE={m:.2f}%")

    out = {
        "n_cells": len(cells),
        "n_seeds": len(per_seed_predictions),
        "overall": {"MAPE": overall_mape, "RMSE": overall_rmse},
        "per_chemistry": chem_results,
        "per_cell": [{"cell_id": c, "chem": ch, "y_true": float(y),
                      "pred_ensemble": float(p)}
                     for c, ch, y, p in zip(cell_ids, chems, ys, ensemble)],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
