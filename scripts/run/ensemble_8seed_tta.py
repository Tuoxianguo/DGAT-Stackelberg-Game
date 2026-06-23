"""
Final ensemble: 8 seeds × N-TTA test-time augmentation for the best deployable
predictor on MIT/Stanford-Toyota 5-fold CV.

TTA: each ckpt makes 1 clean prediction + (N_TTA-1) noisy predictions
(channel-wise Gaussian noise with std = noise_factor × channel_std),
then geom-mean across all TTA samples per ckpt.

Final ensemble: geom-mean (and trimmed mean) across 8 seeds × cells.

Defaults: N_TTA=6, noise_factor=0.01.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sklearn.model_selection import KFold

from battery_paper.data import BSEBenchLoader, BSEEarlyPredictDataset, load_mit_meta
from battery_paper.models.baselines import VanillaTransformerRUL


def _load_vanilla(ckpt_path: Path, device="cuda"):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state["cfg"]
    F_in = len(cfg["features"])
    name = cfg["model"]
    if name == "vanilla":
        model = VanillaTransformerRUL(in_features=F_in, intra_len=cfg["intra_len"],
                                      d_model=cfg["d_model"], n_layers=cfg["n_layers"],
                                      n_heads=cfg["n_heads"], d_aux_feat=0)
    elif name == "lstm":
        from battery_paper.models.baselines import LSTMRUL
        model = LSTMRUL(in_features=F_in, intra_len=cfg["intra_len"],
                        d_model=cfg["d_model"], n_layers=cfg["n_layers"], d_aux_feat=0)
    elif name == "battery_gpt":
        from battery_paper.models.baselines import BatteryGPTLite
        model = BatteryGPTLite(in_features=F_in, intra_len=cfg["intra_len"],
                               d_model=cfg["d_model"], n_layers=cfg["n_layers"],
                               n_heads=cfg["n_heads"], d_aux_feat=0)
    elif name == "pbt":
        from battery_paper.models.baselines import PBTLite
        model = PBTLite(in_features=F_in, intra_len=cfg["intra_len"],
                        d_model=cfg["d_model"], n_layers=cfg["n_layers"],
                        n_heads=cfg["n_heads"], d_aux_feat=0)
    elif name == "dgat":
        from battery_paper.models.baselines import DGATLite
        model = DGATLite(in_features=F_in, intra_len=cfg["intra_len"],
                         d_model=cfg["d_model"], n_layers=cfg["n_layers"],
                         n_heads=cfg["n_heads"], window_size=10, d_aux_feat=0)
    elif name == "dgat_plus":
        from battery_paper.models.baselines import DGATPlusLite
        model = DGATPlusLite(in_features=F_in, intra_len=cfg["intra_len"],
                             d_model=cfg["d_model"], n_layers=cfg["n_layers"],
                             n_heads=cfg["n_heads"], window_size=10, d_aux_feat=0)
    else:
        raise ValueError(f"unknown model {name}")
    model = model.to(device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    return model, cfg


def _predict_tta(model, x, m, n_tta: int = 6, noise_factor: float = 0.01) -> np.ndarray:
    """Return mean prediction across n_tta passes (1 clean + n_tta-1 noisy)."""
    preds_log = []
    with torch.no_grad():
        # clean
        y_clean = model(x, m).cpu().numpy()
        preds_log.append(np.log(np.clip(y_clean, 1, None)))
        # noisy
        std_chan = x.std()
        for i in range(n_tta - 1):
            xn = x + torch.randn_like(x) * noise_factor * std_chan
            y_noisy = model(xn, m).cpu().numpy()
            preds_log.append(np.log(np.clip(y_noisy, 1, None)))
    return np.exp(np.mean(np.stack(preds_log, 0), axis=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,7,2026,1024,100,314,777,2025")
    ap.add_argument("--n_tta", type=int, default=6)
    ap.add_argument("--noise_factor", type=float, default=0.01)
    ap.add_argument("--root", default="experiments/v6_vanilla")
    ap.add_argument("--model_tag", default="vanilla",
                    help="checkpoint folder name pattern: {model_tag}__seedS/fold_F/best.pt")
    ap.add_argument("--out", default="experiments/v6_8seed_tta.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"Seeds: {seeds}")
    print(f"TTA passes per ckpt: {args.n_tta}, noise factor: {args.noise_factor}")

    meta = load_mit_meta("data/interim/mit_meta.csv")
    loader = BSEBenchLoader("data/raw/mit_hf")
    manifest = pd.read_csv("data/interim/mit_manifest_full.csv")
    manifest = manifest.dropna(subset=["cycle_life"])
    manifest = manifest[manifest["cycle_life"] > 100]
    cells_all_unshuffled = manifest["cell_id"].tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # (seed, cell) -> prediction
    per_seed_cell = {}
    truth = {}

    t0 = time.time()
    for seed in seeds:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(cells_all_unshuffled))
        cells_shuffled = [cells_all_unshuffled[i] for i in perm]
        kf = KFold(n_splits=5)
        for fi, (_, te_idx) in enumerate(kf.split(cells_shuffled)):
            test_cells = [cells_shuffled[i] for i in te_idx]
            ckpt = Path(args.root) / f"{args.model_tag}__seed{seed}" / f"fold_{fi}" / "best.pt"
            if not ckpt.exists():
                print(f"  MISSING {ckpt}")
                continue
            model, cfg = _load_vanilla(ckpt, device=device)
            ds = BSEEarlyPredictDataset(loader, test_cells, n_cycles=100,
                                        intra_len=64,
                                        features=cfg["features"],
                                        cache_dir="data/processed/mit_cache_full",
                                        external_meta=meta, augment=False)
            xs, ms, ys = [], [], []
            for i in range(len(ds)):
                r = ds[i]
                xs.append(r["x"]); ms.append(r["mask"]); ys.append(r["y"])
            x = torch.stack(xs).to(device); mk = torch.stack(ms).to(device)
            yhat = _predict_tta(model, x, mk, args.n_tta, args.noise_factor)
            for j, cid in enumerate(test_cells):
                per_seed_cell[(seed, cid)] = float(yhat[j])
                truth[cid] = float(ys[j].item())
        print(f"  seed={seed:5d}: done ({time.time()-t0:.0f}s elapsed)")
        try:
            del model
        except UnboundLocalError:
            pass
        torch.cuda.empty_cache()

    # Per-seed MAPE (with TTA)
    print(f"\n=== Per-seed MAPE (with TTA={args.n_tta}) ===")
    seed_metrics = {}
    for seed in seeds:
        preds = []; trues = []
        for cid, y in truth.items():
            if (seed, cid) in per_seed_cell:
                preds.append(per_seed_cell[(seed, cid)])
                trues.append(y)
        preds = np.array(preds); trues = np.array(trues)
        if len(preds):
            mape = np.mean(np.abs(preds - trues) / np.clip(trues, 1, None)) * 100
            rmse = np.sqrt(np.mean((preds - trues) ** 2))
            seed_metrics[seed] = {"MAPE": float(mape), "RMSE": float(rmse)}
            print(f"  seed={seed:5d}  MAPE={mape:6.2f}%  RMSE={rmse:6.1f}  n={len(preds)}")

    def _ens(combo_fn, label):
        preds_, trues_ = [], []
        for cid, y in truth.items():
            pset = [per_seed_cell[(s, cid)] for s in seeds if (s, cid) in per_seed_cell]
            if not pset:
                continue
            ge = combo_fn(np.array(pset))
            preds_.append(ge); trues_.append(y)
        preds_ = np.array(preds_); trues_ = np.array(trues_)
        mape = np.mean(np.abs(preds_ - trues_) / np.clip(trues_, 1, None)) * 100
        rmse = np.sqrt(np.mean((preds_ - trues_) ** 2))
        print(f"  {label:25s}  MAPE={mape:6.2f}%  RMSE={rmse:6.1f}  n={len(preds_)}")
        return {"MAPE": float(mape), "RMSE": float(rmse), "n": len(preds_)}

    print(f"\n=== {len(seeds)}-seed × TTA ensembles ===")
    geo = _ens(lambda a: float(np.exp(np.mean(np.log(np.clip(a, 1, None))))),
               "Geom mean")
    median = _ens(lambda a: float(np.median(a)), "Median")
    def trimmed(a):
        if len(a) >= 4:
            srt = np.sort(a)
            keep = srt[1:-1]
        else:
            keep = a
        return float(np.exp(np.mean(np.log(np.clip(keep, 1, None)))))
    trim = _ens(trimmed, "Trimmed mean")
    def trim2(a):
        if len(a) >= 6:
            srt = np.sort(a)
            keep = srt[2:-2]
        else:
            keep = a
        return float(np.exp(np.mean(np.log(np.clip(keep, 1, None)))))
    trim2_m = _ens(trim2, "Trimmed mean (drop 2)")
    arith = _ens(lambda a: float(np.mean(a)), "Arithmetic mean")

    # Per-cell prediction record (for diagnostic plots)
    per_cell_records = []
    for cid, y in truth.items():
        pset = [per_seed_cell[(s, cid)] for s in seeds if (s, cid) in per_seed_cell]
        if not pset:
            continue
        arr = np.array(pset)
        median_pred = float(np.median(arr))
        per_cell_records.append({
            "cell_id": cid, "y_true": y,
            "preds": pset,
            "median": median_pred,
            "ape_median": abs(median_pred - y) / y * 100,
            "n_seeds_used": len(pset),
        })

    out = {
        "n_seeds": len(seeds), "n_tta": args.n_tta,
        "noise_factor": args.noise_factor,
        "seed_metrics": seed_metrics,
        "ensemble_geom_mean": geo,
        "ensemble_median": median,
        "ensemble_trimmed_mean_drop1": trim,
        "ensemble_trimmed_mean_drop2": trim2_m,
        "ensemble_arithmetic_mean": arith,
        "per_cell": per_cell_records,
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
