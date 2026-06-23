"""
HUST battery dataset loader (Tian et al. Energy 2022, Zenodo 10.5281/zenodo.6405084).

Three sub-datasets (NCA / NCM / NCM-NCA), each as a folder of CSV files.
Each CSV is one cell with columns:
    time/s, control/V/mA, Ecell/V, <I>/mA, Q discharge/mA.h, Q charge/mA.h,
    control/V, control/mA, cycle number

We aggregate per-cycle:
    cycle, t (per cycle relative), V, I (A), Q_dis (mAh), Q_chg (mAh)

Cycle_life is computed as the cycle index where Q_discharge drops below
0.8 * Q_initial (80% SOH).

We build the same tensor format as BSEEarlyPredictDataset:
    x: (N_cycles, F=4 channels [V, I, T_synth, Q_synth], L=64 intra)
    mask, proto (placeholder [1, 80, 1]), y
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ..utils.logging_utils import get_logger

LOG = get_logger("hust_loader")

# Three HUST subsets and nominal capacities (mAh)
SUBSETS = {
    "Dataset_1_NCA_battery": {"chem": "NCA", "Q_nom_mAh": 3500},
    "Dataset_2_NCM_battery": {"chem": "NCM", "Q_nom_mAh": 3500},
    "Dataset_3_NCM_NCA_battery": {"chem": "NCM_NCA", "Q_nom_mAh": 3500},
}


def _load_csv(p: Path) -> pd.DataFrame | None:
    """Load one HUST CSV. Returns clean df with renamed cols."""
    try:
        df = pd.read_csv(p)
    except Exception as e:
        LOG.warning("read %s failed: %s", p.name, e)
        return None
    # Normalize column names
    col_map = {
        "time/s":           "time_s",
        "Ecell/V":          "voltage_v",
        "<I>/mA":           "current_mA",
        "Q discharge/mA.h": "q_dis_mAh",
        "Q charge/mA.h":    "q_chg_mAh",
        "cycle number":     "cycle",
    }
    keep_cols = [c for c in col_map if c in df.columns]
    if not keep_cols:
        return None
    df = df[keep_cols].rename(columns=col_map)
    # Convert mA -> A
    df["current_a"] = df["current_mA"] / 1000.0
    # Cumulative capacity in Ah (use discharge - charge as signed Q)
    df["capacity_ah"] = (df["q_chg_mAh"] - df["q_dis_mAh"]) / 1000.0
    df["cycle"] = df["cycle"].astype(int)
    return df


def _compute_cycle_life(df: pd.DataFrame, q_nom_mAh: float = 3500,
                       eol_frac: float = 0.80) -> int:
    """Compute cycle_life: first cycle where max Q_discharge < 80% of initial."""
    per_cycle_qd = df.groupby("cycle")["q_dis_mAh"].max()
    if per_cycle_qd.empty:
        return 0
    q_init = float(per_cycle_qd.iloc[: min(5, len(per_cycle_qd))].max())
    thresh = eol_frac * q_init
    bad = per_cycle_qd[per_cycle_qd < thresh]
    if len(bad) == 0:
        return int(per_cycle_qd.index.max())
    return int(bad.index[0])


def _build_tensor(df: pd.DataFrame, n_cycles: int = 100,
                  intra_len: int = 64) -> dict | None:
    """Build (N, F=4, L) tensor from per-cycle data."""
    cycles = sorted(df["cycle"].unique())[:n_cycles]
    if len(cycles) < 10:
        return None
    F = 4
    x = np.zeros((n_cycles, F, intra_len), dtype=np.float32)
    mask = np.zeros(n_cycles, dtype=np.float32)
    for i, cy in enumerate(cycles):
        sub = df[df["cycle"] == cy]
        if len(sub) < 4:
            continue
        t = sub["time_s"].values
        if t.max() - t.min() <= 0:
            continue
        t_norm = (t - t.min()) / (t.max() - t.min())
        grid = np.linspace(0, 1, intra_len)
        # Channels: V, I, T_synth (25°C), capacity_Ah
        x[i, 0, :] = np.interp(grid, t_norm, sub["voltage_v"].values).astype(np.float32)
        x[i, 1, :] = np.interp(grid, t_norm, sub["current_a"].values).astype(np.float32)
        x[i, 2, :] = 25.0   # HUST didn't log temperature; synth 25°C
        x[i, 3, :] = np.interp(grid, t_norm, sub["capacity_ah"].values).astype(np.float32)
        mask[i] = 1.0
    np.nan_to_num(x, copy=False, nan=0.0)
    return dict(x=x, mask=mask)


def load_hust_cells(root: str, n_cycles: int = 100, intra_len: int = 64,
                    max_cells_per_subset: int | None = None) -> List[dict]:
    """Return list of cell dicts: x, mask, y (cycle_life), chem."""
    root = Path(root)
    cells = []
    for subset, info in SUBSETS.items():
        sub_dir = root / subset
        if not sub_dir.exists():
            LOG.warning("missing %s", sub_dir)
            continue
        csvs = sorted(sub_dir.glob("*.csv"))
        if max_cells_per_subset:
            csvs = csvs[:max_cells_per_subset]
        for csv in csvs:
            df = _load_csv(csv)
            if df is None:
                continue
            cl = _compute_cycle_life(df, q_nom_mAh=info["Q_nom_mAh"])
            if cl < 50:
                LOG.warning("%s cycle_life=%d too small, skip", csv.name, cl)
                continue
            t = _build_tensor(df, n_cycles=n_cycles, intra_len=intra_len)
            if t is None:
                continue
            t["y"] = float(cl)
            t["cell_id"] = csv.stem
            t["chem"] = info["chem"]
            # Placeholder protocol (HUST uses different conventions, no CC1/CC2)
            t["proto"] = np.array([1.0, 80.0, 1.0], dtype=np.float32)
            cells.append(t)
            LOG.info("loaded %s/%s: cycle_life=%d, valid_cycles=%d",
                     subset, csv.stem, cl, int(t["mask"].sum()))
    return cells


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/hust")
    ap.add_argument("--max_per_subset", type=int, default=5)
    args = ap.parse_args()
    cells = load_hust_cells(args.root, max_cells_per_subset=args.max_per_subset)
    print(f"\nLoaded {len(cells)} HUST cells")
    for c in cells[:10]:
        print(f"  {c['cell_id']:25s} chem={c['chem']:8s} y={c['y']:.0f} "
              f"valid={int(c['mask'].sum())}")
