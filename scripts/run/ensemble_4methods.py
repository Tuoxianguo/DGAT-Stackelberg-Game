"""
Cross-method ensemble: combine 4 methods × 3 seeds = 12 models.
Methods: vanilla, dgat, pbt, lstm (drop battery_gpt as it's worst).

Expected: median ensemble breaks 9% MAPE.
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
from battery_paper.models.baselines import (VanillaTransformerRUL, LSTMRUL,
                                            BatteryGPTLite, PBTLite, DGATLite)

CKPT_ROOTS = {
    "vanilla":     "experiments/v6_vanilla",
    "lstm":        "experiments/v6_lstm",
    "battery_gpt": "experiments/v6_battery_gpt",
    "pbt":         "experiments/v6_pbt",
    "dgat":        "experiments/v6_dgat",
}
SEEDS = [42, 7, 2026]


def _load(ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state["cfg"]
    F_in = len(cfg["features"])
    name = cfg["model"]
    if name == "vanilla":
        m = VanillaTransformerRUL(in_features=F_in, intra_len=cfg["intra_len"],
                                  d_model=cfg["d_model"], n_layers=cfg["n_layers"],
                                  n_heads=cfg["n_heads"], d_aux_feat=0)
    elif name == "lstm":
        m = LSTMRUL(in_features=F_in, intra_len=cfg["intra_len"],
                    d_model=cfg["d_model"], n_layers=cfg["n_layers"], d_aux_feat=0)
    elif name == "battery_gpt":
        m = BatteryGPTLite(in_features=F_in, intra_len=cfg["intra_len"],
                           d_model=cfg["d_model"], n_layers=cfg["n_layers"],
                           n_heads=cfg["n_heads"], d_aux_feat=0)
    elif name == "pbt":
        m = PBTLite(in_features=F_in, intra_len=cfg["intra_len"],
                    d_model=cfg["d_model"], n_layers=cfg["n_layers"],
                    n_heads=cfg["n_heads"], d_aux_feat=0)
    elif name == "dgat":
        m = DGATLite(in_features=F_in, intra_len=cfg["intra_len"],
                     d_model=cfg["d_model"], n_layers=cfg["n_layers"],
                     n_heads=cfg["n_heads"], window_size=10, d_aux_feat=0)
    else:
        raise ValueError(name)
    m = m.to(device)
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

    # (method, seed, cell_id) -> pred
    all_preds = {}
    truth = {}

    for method, root in CKPT_ROOTS.items():
        root = Path(root)
        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            perm = rng.permutation(len(cells_all_unshuffled))
            cells_shuffled = [cells_all_unshuffled[i] for i in perm]
            kf = KFold(n_splits=5)
            for fi, (_, te_idx) in enumerate(kf.split(cells_shuffled)):
                test_cells = [cells_shuffled[i] for i in te_idx]
                ckpt = root / f"{method}__seed{seed}" / f"fold_{fi}" / "best.pt"
                if not ckpt.exists():
                    continue
                m, cfg = _load(ckpt, device=device)
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
                with torch.no_grad():
                    # TTA: 1 clean + 4 noisy
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

    # Per-method ensemble
    print("\n=== Per-method 3-seed × 5-TTA ENSEMBLE (median) ===")
    per_method = {}
    for method in CKPT_ROOTS:
        preds = []; trues = []
        for cid, y in truth.items():
            pset = [all_preds[(method, s, cid)] for s in SEEDS
                    if (method, s, cid) in all_preds]
            if not pset:
                continue
            preds.append(float(np.median(pset))); trues.append(y)
        if not preds:
            continue
        preds = np.array(preds); trues = np.array(trues)
        mape = float(np.mean(np.abs(preds - trues) / np.clip(trues, 1, None)) * 100)
        rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
        per_method[method] = {"MAPE": mape, "RMSE": rmse}
        print(f"  {method:15s}  MAPE={mape:.2f}%  RMSE={rmse:.1f}")

    # Cross-method 4-method × 3-seed = 12-model ensemble (drop battery_gpt)
    print("\n=== Cross-method (vanilla+dgat+pbt+lstm) × 3 seeds × TTA ===")
    methods_use = ["vanilla", "dgat", "pbt", "lstm"]
    for combo_name, combo_fn in [
        ("geom_mean", lambda a: float(np.exp(np.mean(np.log(np.clip(a, 1, None)))))),
        ("median",    lambda a: float(np.median(a))),
        ("trimmed (drop 2)", lambda a: float(np.exp(np.mean(np.log(np.clip(
            np.sort(a)[1:-1] if len(a) >= 4 else a, 1, None)))))),
    ]:
        preds = []; trues = []
        for cid, y in truth.items():
            pset = [all_preds[(m, s, cid)]
                    for m in methods_use for s in SEEDS
                    if (m, s, cid) in all_preds]
            if not pset:
                continue
            preds.append(combo_fn(np.array(pset))); trues.append(y)
        preds = np.array(preds); trues = np.array(trues)
        mape = float(np.mean(np.abs(preds - trues) / np.clip(trues, 1, None)) * 100)
        rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
        print(f"  {combo_name:20s}  MAPE={mape:.2f}%  RMSE={rmse:.1f}  n={len(preds)}")

    # Save
    json.dump({"per_method": per_method, "n_seeds": len(SEEDS)},
              open("experiments/v7_4method_ensemble.json", "w"),
              indent=2)


if __name__ == "__main__":
    main()
