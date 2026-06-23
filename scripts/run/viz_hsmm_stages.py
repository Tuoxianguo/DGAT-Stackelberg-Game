"""
Visualise the HSMM stage posterior γ_t(k) along cycles for a few representative
cells (long-life / median / short-life). Loads the trained Full model checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from battery_paper.data import BSEBenchLoader, BSEEarlyPredictDataset, load_mit_meta
from battery_paper.models.proposed import HSMMGraphGameModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to best.pt of a full HSMM-GraphGame run")
    ap.add_argument("--data_root", default="data/raw/mit_hf")
    ap.add_argument("--meta_csv", default="data/interim/mit_meta.csv")
    ap.add_argument("--cache_dir", default="data/processed/mit_cache_full")
    ap.add_argument("--out_png", default="experiments/viz_hsmm_stages.png")
    ap.add_argument("--n_cells", type=int, default=6)
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = state["cfg"]
    model = HSMMGraphGameModel(
        in_features=len(cfg["features"]), intra_len=cfg["intra_len"],
        d_model=cfg["d_model"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
        hsmm_K=cfg["hsmm_K"], hsmm_D_max=cfg["hsmm_D_max"],
        use_graph=cfg["use_graph"],
    )
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    meta = load_mit_meta(args.meta_csv)
    loader = BSEBenchLoader(args.data_root)
    all_cells = loader.list_cells()
    cells_sorted = sorted(all_cells,
                          key=lambda c: meta.get(c, {}).get("cycle_life", 0))
    # pick short / median / long
    n = args.n_cells
    idxs = np.linspace(0, len(cells_sorted) - 1, n).astype(int)
    pick = [cells_sorted[i] for i in idxs]
    ys = [meta.get(c, {}).get("cycle_life", "?") for c in pick]

    ds = BSEEarlyPredictDataset(loader, pick, n_cycles=cfg["n_cycles"],
                                intra_len=cfg["intra_len"],
                                features=cfg["features"],
                                cache_dir=cfg["cache_dir"],
                                external_meta=meta, augment=False)

    fig, axes = plt.subplots(n, 1, figsize=(10, 1.7 * n), sharex=True)
    if n == 1:
        axes = [axes]
    with torch.no_grad():
        for i, ax in enumerate(axes):
            rec = ds[i]
            out = model(rec["x"].unsqueeze(0), rec["mask"].unsqueeze(0),
                        rec["proto"].unsqueeze(0))
            gamma = out.stage_post[0].numpy()         # (T, K)
            mask = rec["mask"].numpy()
            T = int(mask.sum())
            gamma = gamma[:T]
            ax.imshow(gamma.T, aspect="auto", origin="lower",
                      vmin=0, vmax=1, cmap="viridis")
            ax.set_yticks(range(cfg["hsmm_K"]))
            ax.set_yticklabels([f"S{k}" for k in range(cfg["hsmm_K"])])
            ax.set_title(f"{pick[i]}  cycle_life={ys[i]}  RUL_hat={out.rul_hat.item():.0f}")
    axes[-1].set_xlabel("Cycle (early data)")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=150)
    plt.close()
    print(f"saved {args.out_png}")


if __name__ == "__main__":
    main()
