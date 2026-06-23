"""
12-model ensemble: combine 4 seeds × 3 models (vanilla, graph, full) = 12 ckpts.
This gives us the absolute best deployable predictor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sklearn.model_selection import KFold

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


def _predict(model, x, m, p):
    with torch.no_grad():
        if isinstance(model, VanillaTransformerRUL):
            return model(x, m)
        else:
            out = model(x, m, p)
            return out.rul_hat


def main():
    seeds = [42, 7, 2026, 1024]
    runs = {
        "vanilla": "experiments/v6_vanilla",
        "graph_only": "experiments/v6_final",
        "full": "experiments/v6_full",
    }

    meta = load_mit_meta("data/interim/mit_meta.csv")
    loader = BSEBenchLoader("data/raw/mit_hf")
    manifest = pd.read_csv("data/interim/mit_manifest_full.csv")
    manifest = manifest.dropna(subset=["cycle_life"])
    manifest = manifest[manifest["cycle_life"] > 100]
    cells_all_unshuffled = manifest["cell_id"].tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # (model_tag, seed, cell_id) -> pred
    all_preds = {}
    truth = {}

    for model_tag, root in runs.items():
        root = Path(root)
        for seed in seeds:
            rng = np.random.RandomState(seed)
            perm = rng.permutation(len(cells_all_unshuffled))
            cells_shuffled = [cells_all_unshuffled[i] for i in perm]
            kf = KFold(n_splits=5)
            for fi, (_, te_idx) in enumerate(kf.split(cells_shuffled)):
                test_cells = [cells_shuffled[i] for i in te_idx]
                ckpt = root / f"{model_tag}__seed{seed}" / f"fold_{fi}" / "best.pt"
                if not ckpt.exists():
                    continue
                model, cfg = _load_model(ckpt, device=device)
                ds = BSEEarlyPredictDataset(loader, test_cells, n_cycles=100,
                                            intra_len=64,
                                            features=cfg["features"],
                                            cache_dir="data/processed/mit_cache_full",
                                            external_meta=meta, augment=False)
                xs = []; ms = []; ps = []; ys = []
                for i in range(len(ds)):
                    r = ds[i]
                    xs.append(r["x"]); ms.append(r["mask"])
                    ps.append(r["proto"]); ys.append(r["y"])
                x = torch.stack(xs).to(device); mk = torch.stack(ms).to(device)
                p = torch.stack(ps).to(device)
                yhat = _predict(model, x, mk, p).cpu().numpy()
                for j, cid in enumerate(test_cells):
                    all_preds[(model_tag, seed, cid)] = float(yhat[j])
                    truth[cid] = float(ys[j].item())
            del model
            torch.cuda.empty_cache()

    # 12-model ensemble per cell
    print("\n=== Per-model 4-seed geo-mean ENSEMBLE ===")
    per_model_ens = {}
    for model_tag in runs:
        preds = []; trues = []
        for cid, y in truth.items():
            pset = [all_preds[(model_tag, s, cid)] for s in seeds
                    if (model_tag, s, cid) in all_preds]
            if not pset:
                continue
            ge = float(np.exp(np.mean(np.log(np.clip(np.array(pset), 1, None)))))
            preds.append(ge); trues.append(y)
        preds = np.array(preds); trues = np.array(trues)
        mape = np.mean(np.abs(preds - trues) / np.clip(trues, 1, None)) * 100
        rmse = np.sqrt(np.mean((preds - trues) ** 2))
        per_model_ens[model_tag] = {"MAPE": float(mape), "RMSE": float(rmse),
                                     "n": len(preds)}
        print(f"  {model_tag:12s}  MAPE={mape:.2f}%  RMSE={rmse:.1f}  n={len(preds)}")

    print("\n=== 12-model cross-method ENSEMBLE (4 seeds × 3 models) ===")
    preds_all = []; trues_all = []
    for cid, y in truth.items():
        pset = [all_preds[(mt, s, cid)] for mt in runs for s in seeds
                if (mt, s, cid) in all_preds]
        if not pset:
            continue
        ge = float(np.exp(np.mean(np.log(np.clip(np.array(pset), 1, None)))))
        preds_all.append(ge); trues_all.append(y)
    preds_all = np.array(preds_all); trues_all = np.array(trues_all)
    mape = np.mean(np.abs(preds_all - trues_all) / np.clip(trues_all, 1, None)) * 100
    rmse = np.sqrt(np.mean((preds_all - trues_all) ** 2))
    print(f"  12-model    MAPE={mape:.2f}%  RMSE={rmse:.1f}  n={len(preds_all)}")

    print("\n=== TRIMMED ensemble (drop best+worst across 12) ===")
    preds_tr = []; trues_tr = []
    for cid, y in truth.items():
        pset = sorted([all_preds[(mt, s, cid)] for mt in runs for s in seeds
                       if (mt, s, cid) in all_preds])
        if len(pset) < 3:
            continue
        keep = pset[1:-1]
        tm = float(np.exp(np.mean(np.log(np.clip(np.array(keep), 1, None)))))
        preds_tr.append(tm); trues_tr.append(y)
    preds_tr = np.array(preds_tr); trues_tr = np.array(trues_tr)
    mape_t = np.mean(np.abs(preds_tr - trues_tr) / np.clip(trues_tr, 1, None)) * 100
    rmse_t = np.sqrt(np.mean((preds_tr - trues_tr) ** 2))
    print(f"  Trimmed     MAPE={mape_t:.2f}%  RMSE={rmse_t:.1f}")

    # Weighted ensemble: vanilla weight 2 (since it's the best single)
    print("\n=== Weighted: vanilla=2x weight + graph=1x + full=1x ===")
    preds_w = []; trues_w = []
    for cid, y in truth.items():
        van = [all_preds[("vanilla", s, cid)] for s in seeds
               if ("vanilla", s, cid) in all_preds]
        gra = [all_preds[("graph_only", s, cid)] for s in seeds
               if ("graph_only", s, cid) in all_preds]
        ful = [all_preds[("full", s, cid)] for s in seeds
               if ("full", s, cid) in all_preds]
        if not van:
            continue
        # log-mean with weights
        all_p = (np.log(np.array(van + van) + 1).tolist()  # 2x weight
                 + np.log(np.array(gra) + 1).tolist()
                 + np.log(np.array(ful) + 1).tolist())
        if not all_p:
            continue
        wm = float(np.exp(np.mean(all_p)))
        preds_w.append(wm); trues_w.append(y)
    preds_w = np.array(preds_w); trues_w = np.array(trues_w)
    mape_w = np.mean(np.abs(preds_w - trues_w) / np.clip(trues_w, 1, None)) * 100
    rmse_w = np.sqrt(np.mean((preds_w - trues_w) ** 2))
    print(f"  Weighted    MAPE={mape_w:.2f}%  RMSE={rmse_w:.1f}")

    out = {
        "per_model_ensemble": per_model_ens,
        "all_12_geomean":     {"MAPE": float(mape), "RMSE": float(rmse)},
        "all_12_trimmed":     {"MAPE": float(mape_t), "RMSE": float(rmse_t)},
        "weighted_vanilla2x": {"MAPE": float(mape_w), "RMSE": float(rmse_w)},
    }
    json.dump(out, open("experiments/v6_12model_ensemble.json", "w"), indent=2)
    print(f"\nSaved to experiments/v6_12model_ensemble.json")


if __name__ == "__main__":
    main()
