"""
Cross-architecture ensemble: Vanilla CT 6 seeds + DGAT 6 seeds = 12 models,
each × 5-TTA. Hopefully breaks 9% MAPE through architectural diversity.
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
from battery_paper.models.baselines import VanillaTransformerRUL, DGATLite

CONFIGS = [
    ("vanilla", "experiments/v6_vanilla", VanillaTransformerRUL,
     [42, 7, 2026, 1024, 100, 777]),
    ("dgat",    "experiments/v6_dgat",    DGATLite,
     [42, 7, 2026, 100, 777, 1024]),
]


def _load(ckpt_path, cls, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state["cfg"]
    F_in = len(cfg["features"])
    kw = dict(in_features=F_in, intra_len=cfg["intra_len"],
              d_model=cfg["d_model"], n_layers=cfg["n_layers"],
              n_heads=cfg["n_heads"], d_aux_feat=0)
    if cls is DGATLite:
        kw["window_size"] = 10
    m = cls(**kw).to(device)
    m.load_state_dict(state["model"], strict=False)
    m.eval()
    return m, cfg


def main():
    meta = load_mit_meta("data/interim/mit_meta.csv")
    loader = BSEBenchLoader("data/raw/mit_hf")
    manifest = pd.read_csv("data/interim/mit_manifest_full.csv")
    manifest = manifest.dropna(subset=["cycle_life"])
    manifest = manifest[manifest["cycle_life"] > 100]
    cells_all_unshuffled = manifest["cell_id"].tolist()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # (method, seed, cell) -> pred
    all_preds = {}
    truth = {}

    for method, root, cls, seeds in CONFIGS:
        root = Path(root)
        for seed in seeds:
            rng = np.random.RandomState(seed)
            perm = rng.permutation(len(cells_all_unshuffled))
            cells_shuffled = [cells_all_unshuffled[i] for i in perm]
            kf = KFold(n_splits=5)
            for fi, (_, te_idx) in enumerate(kf.split(cells_shuffled)):
                test_cells = [cells_shuffled[i] for i in te_idx]
                ckpt = root / f"{method}__seed{seed}" / f"fold_{fi}" / "best.pt"
                if not ckpt.exists():
                    continue
                m, cfg = _load(ckpt, cls, device=device)
                ds = BSEEarlyPredictDataset(loader, test_cells, n_cycles=100,
                                            intra_len=64,
                                            features=cfg["features"],
                                            cache_dir="data/processed/mit_cache_full",
                                            external_meta=meta, augment=False)
                xs = []; ms = []; ys = []
                for i in range(len(ds)):
                    r = ds[i]
                    xs.append(r["x"]); ms.append(r["mask"]); ys.append(r["y"])
                x = torch.stack(xs).to(device); mk = torch.stack(ms).to(device)
                # TTA: 1 clean + 4 noisy
                with torch.no_grad():
                    preds_log = [np.log(np.clip(m(x, mk).cpu().numpy(), 1, None))]
                    for _ in range(4):
                        xn = x + torch.randn_like(x) * 0.003 * x.std()
                        preds_log.append(np.log(np.clip(m(xn, mk).cpu().numpy(), 1, None)))
                    yhat = np.exp(np.mean(np.stack(preds_log, 0), 0))
                for j, cid in enumerate(test_cells):
                    all_preds[(method, seed, cid)] = float(yhat[j])
                    truth[cid] = float(ys[j].item())
                del m
            torch.cuda.empty_cache()
        print(f"  {method}: done")

    # Per-method 6-seed × 5-TTA
    print("\n=== Per-method 6-seed × 5-TTA ENSEMBLE ===")
    for method, _, _, seeds in CONFIGS:
        for label, fn in [("Geom", lambda a: float(np.exp(np.mean(np.log(np.clip(a, 1, None)))))),
                          ("Median", lambda a: float(np.median(a))),
                          ("Trimmed(drop 2)", lambda a: float(np.exp(np.mean(np.log(np.clip(
                              np.sort(a)[1:-1] if len(a) >= 4 else a, 1, None))))))]:
            preds = []; trues = []
            for cid, y in truth.items():
                pset = [all_preds[(method, s, cid)] for s in seeds
                        if (method, s, cid) in all_preds]
                if not pset:
                    continue
                preds.append(fn(np.array(pset))); trues.append(y)
            preds = np.array(preds); trues = np.array(trues)
            mape = np.mean(np.abs(preds - trues) / np.clip(trues, 1, None)) * 100
            rmse = np.sqrt(np.mean((preds - trues) ** 2))
            print(f"  {method:8s} {label:15s}  MAPE={mape:.2f}%  RMSE={rmse:.1f}")

    # Cross-architecture ensemble: 12 models × 5-TTA
    print("\n=== CT + DGAT 12-seed × 5-TTA Cross-Architecture ENSEMBLE ===")
    for label, fn in [("Geom", lambda a: float(np.exp(np.mean(np.log(np.clip(a, 1, None)))))),
                      ("Median", lambda a: float(np.median(a))),
                      ("Trimmed(drop 2)", lambda a: float(np.exp(np.mean(np.log(np.clip(
                          np.sort(a)[1:-1] if len(a) >= 4 else a, 1, None))))))]:
        preds = []; trues = []
        for cid, y in truth.items():
            pset = [all_preds[(m, s, cid)] for m, _, _, sd in CONFIGS for s in sd
                    if (m, s, cid) in all_preds]
            if not pset:
                continue
            preds.append(fn(np.array(pset))); trues.append(y)
        preds = np.array(preds); trues = np.array(trues)
        mape = np.mean(np.abs(preds - trues) / np.clip(trues, 1, None)) * 100
        rmse = np.sqrt(np.mean((preds - trues) ** 2))
        print(f"  CT+DGAT  {label:15s}  MAPE={mape:.2f}%  RMSE={rmse:.1f}  n={len(preds)}")

    json.dump({"per_cell": [{"cell_id": c, "y_true": y} for c, y in truth.items()]},
              open("experiments/v8_ct_dgat_ensemble.json", "w"),
              indent=2)


if __name__ == "__main__":
    main()
