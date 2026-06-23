"""
Cross-domain evaluation: load trained models (per-fold checkpoints from
sweep_v3_5fold) and evaluate on CALCE cells with the same input pipeline.

We use the MIT-trained model with its (V/I/T/Q) channel ordering directly on
CALCE data (which only has V and I; we synthesise T = 25C constant and
Q = cumulative integral). This is a zero-shot transfer evaluation.

Outputs CALCE per-cell predictions and aggregate MAPE.
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

from battery_paper.data.calce_loader import load_all_calce
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


def _calce_to_full_features(calce_cell, intra_len=64):
    """CALCE only has (V, I); pad to (V, I, T_synth, Q_synth) to match MIT model."""
    x_2 = calce_cell["x"]              # (N, 2, L)  with features=("voltage","current")
    N, _, L = x_2.shape
    T_syn = np.full((N, 1, L), 25.0, dtype=np.float32)
    # Q_synth = cumulative integral of |I| along intra-cycle axis
    I = x_2[:, 1:2, :]                 # (N, 1, L)
    Q_syn = np.cumsum(np.abs(I), axis=-1) / L * 0.3   # rough scaling
    x_4 = np.concatenate([x_2, T_syn, Q_syn], axis=1).astype(np.float32)
    return x_4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_root", default="experiments/sweep_v3_5fold")
    ap.add_argument("--calce_root", default="data/raw/calce")
    ap.add_argument("--out", default="experiments/cross_domain_calce.json")
    args = ap.parse_args()

    # Load CALCE
    cells = load_all_calce(args.calce_root, n_cycles=100, intra_len=64)
    print(f"loaded {len(cells)} CALCE cells")
    if not cells:
        print("NO CALCE CELLS")
        return
    for c in cells:
        c["x_full"] = _calce_to_full_features(c)
        c["y_true"] = (c["y"].item() if hasattr(c["y"], "item") else float(c["y"]))
        # fallback to published cycle_life if our extraction gave 1
        if c["y_true"] <= 1:
            from battery_paper.data.calce_loader import CELL_INFO
            c["y_true"] = float(CELL_INFO.get(c["cell_id"], {})
                                .get("cycle_life_true", 500))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sweep_root = Path(args.sweep_root)
    models = ["vanilla", "graph_only", "hsmm_graph", "full"]
    summary = {}
    for m_name in models:
        run_dir = sweep_root / f"{m_name}__5fold"
        if not run_dir.exists():
            continue
        # ensemble across folds
        all_preds = []
        for fold_dir in sorted(run_dir.glob("fold_*")):
            ckpt = fold_dir / "best.pt"
            if not ckpt.exists():
                continue
            model, cfg = _load_model(ckpt, device=device)
            preds_per_cell = []
            with torch.no_grad():
                for c in cells:
                    x = torch.tensor(c["x_full"]).unsqueeze(0).to(device)
                    mk = torch.tensor(c["mask"]).unsqueeze(0).to(device)
                    p = torch.tensor(c["proto"]).unsqueeze(0).to(device)
                    if isinstance(model, VanillaTransformerRUL):
                        yhat = model(x, mk)
                    else:
                        out = model(x, mk, p)
                        yhat = out.rul_hat
                    preds_per_cell.append(float(yhat.cpu().numpy()[0]))
            all_preds.append(preds_per_cell)
        if not all_preds:
            continue
        all_preds = np.array(all_preds)   # (n_folds, n_cells)
        ens = all_preds.mean(axis=0)
        true = np.array([c["y_true"] for c in cells])
        mape = float(np.mean(np.abs(ens - true) / np.clip(true, 1, None)) * 100)
        rmse = float(np.sqrt(np.mean((ens - true) ** 2)))
        per_cell = [{"cell_id": c["cell_id"], "y_true": float(t),
                     "pred_ens": float(p),
                     "preds_per_fold": [float(x) for x in fp]}
                    for c, t, p, fp in zip(cells, true, ens, all_preds.T)]
        summary[m_name] = {"MAPE": mape, "RMSE": rmse, "n": len(true),
                            "per_cell": per_cell}
        print(f"  {m_name:15s}  MAPE={mape:6.2f}%   RMSE={rmse:6.1f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print("Saved to", args.out)


if __name__ == "__main__":
    main()
