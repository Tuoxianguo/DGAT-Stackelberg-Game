"""Inspect the tensor returned by BSEEarlyPredictDataset for one cell."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from battery_paper.data import BSEBenchLoader, BSEEarlyPredictDataset

loader = BSEBenchLoader("data/raw/mit_hf")
ds = BSEEarlyPredictDataset(loader, loader.list_cells()[:3], n_cycles=50,
                            intra_len=32, cache_dir=None)
for i in range(3):
    rec = ds[i]
    x = rec["x"].numpy(); m = rec["mask"].numpy(); p = rec["proto"].numpy(); y = rec["y"].item()
    print(f"\ncell {ds.cell_ids[i]}: x={x.shape}, mask sum={m.sum():.0f}, y={y}, proto={p}")
    for f in range(x.shape[1]):
        print(f"  feature {f}: min={np.nanmin(x[:, f, :]):.2f}, max={np.nanmax(x[:, f, :]):.2f}, "
              f"mean={np.nanmean(x[:, f, :]):.2f}, nan_count={np.isnan(x[:, f, :]).sum()}, "
              f"inf_count={np.isinf(x[:, f, :]).sum()}")
