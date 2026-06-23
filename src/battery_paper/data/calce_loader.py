"""
CALCE CS2/CX2 lithium-ion battery dataset loader.
Source: https://web.calce.umd.edu/batteries/data.htm

Each cell ZIP contains many .xlsx files (one per test session).
Each xlsx has multiple sheets ("Channel_1-008", "Channel_1-009", etc.).
We aggregate all sessions into a unified per-cycle time-series.

For our cross-domain evaluation we extract a similar tensor shape
(n_cycles, F, intra_len) as the BSEEarlyPredictDataset.

This loader caches the result as .npz per cell.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from ..utils.logging_utils import get_logger

LOG = get_logger("calce_loader")


# CALCE cells, nominal capacity Ah (CS2 = LCO/Graphite 1.1 Ah, CX2 = LCO 1.35 Ah)
# cycle_life_true is the published end-of-life cycle at 80 % SOH, used as
# ground truth when we don't load enough data to detect EOL ourselves.
CELL_INFO = {
    "CS2_35": {"chem": "CS2", "Q_nom": 1.1,  "EOL_frac": 0.80, "cycle_life_true": 822},
    "CS2_36": {"chem": "CS2", "Q_nom": 1.1,  "EOL_frac": 0.80, "cycle_life_true": 818},
    "CS2_37": {"chem": "CS2", "Q_nom": 1.1,  "EOL_frac": 0.80, "cycle_life_true": 1015},
    "CS2_38": {"chem": "CS2", "Q_nom": 1.1,  "EOL_frac": 0.80, "cycle_life_true": 1011},
    "CX2_35": {"chem": "CX2", "Q_nom": 1.35, "EOL_frac": 0.80, "cycle_life_true": 720},
    "CX2_38": {"chem": "CX2", "Q_nom": 1.35, "EOL_frac": 0.80, "cycle_life_true": 1050},
}

# CALCE xlsx column names (varying across files); we map common variants
COL_ALIAS = {
    "voltage": ["Voltage(V)", "V", "Voltage", "Cell Voltage(V)"],
    "current": ["Current(A)", "A", "Current", "Cell Current(A)"],
    "test_time": ["Test_Time(s)", "Test Time(s)", "Test_Time", "TestTime(s)"],
    "step_time": ["Step_Time(s)", "Step Time(s)", "Step_Time"],
    "datetime": ["Date_Time", "DateTime"],
    "cycle": ["Cycle_Index", "Cycle Index", "Cycle"],
    "step_index": ["Step_Index", "Step Index"],
    "charge_capacity": ["Charge_Capacity(Ah)", "Charge Capacity(Ah)"],
    "discharge_capacity": ["Discharge_Capacity(Ah)", "Discharge Capacity(Ah)"],
}


def _find_col(df: pd.DataFrame, alias_key: str) -> Optional[str]:
    for name in COL_ALIAS.get(alias_key, []):
        if name in df.columns:
            return name
    return None


def _read_one_excel(p: Path) -> pd.DataFrame | None:
    try:
        # Use openpyxl for .xlsx; CALCE files have a single sheet typically
        all_sheets = pd.read_excel(p, sheet_name=None, engine="openpyxl")
        # concat all sheets
        dfs = [s for s in all_sheets.values() if len(s)]
        if not dfs:
            return None
        df = pd.concat(dfs, ignore_index=True)
        return df
    except Exception as e:
        LOG.warning("failed reading %s: %s", p.name, e)
        return None


def load_calce_cell(cell_dir: Path, cell_name: str,
                    max_xlsx: int = 8) -> dict | None:
    """Aggregate first `max_xlsx` .xlsx in cell_dir (sorted by date) into a
    per-cycle summary. Loading every xlsx is too slow and we only need
    early-cycle data anyway.

    Returns dict with keys: cycle (np), Qd (np), Qc (np), cycle_life, raw_df
    or None if no usable data.
    """
    xlsx = sorted(cell_dir.glob("**/*.xlsx"))[:max_xlsx]
    if not xlsx:
        LOG.warning("no xlsx in %s", cell_dir)
        return None
    frames = []
    for p in xlsx:
        df = _read_one_excel(p)
        if df is not None and len(df) > 100:
            frames.append(df)
    if not frames:
        return None
    big = pd.concat(frames, ignore_index=True)
    # Required cols
    col_v = _find_col(big, "voltage")
    col_i = _find_col(big, "current")
    col_t = _find_col(big, "test_time")
    col_cy = _find_col(big, "cycle")
    col_qc = _find_col(big, "charge_capacity")
    col_qd = _find_col(big, "discharge_capacity")
    if not all([col_v, col_i, col_cy]):
        LOG.warning("cell %s: missing required columns; have %s",
                    cell_name, list(big.columns)[:10])
        return None
    # Per-cycle aggregates
    big = big.dropna(subset=[col_v, col_cy])
    if col_t and col_t in big.columns:
        big = big.sort_values(col_t)
    grp = big.groupby(col_cy)
    cyc = []; qd = []; qc = []; tavg = []; vmax = []; vmin = []
    for cy, g in grp:
        if len(g) < 5:
            continue
        cyc.append(int(cy))
        qd.append(float(g[col_qd].max()) if col_qd else float("nan"))
        qc.append(float(g[col_qc].max()) if col_qc else float("nan"))
        vmax.append(float(g[col_v].max()))
        vmin.append(float(g[col_v].min()))
        tavg.append(float("nan"))
    cyc = np.asarray(cyc); qd = np.asarray(qd); qc = np.asarray(qc)
    if not len(cyc):
        return None
    # Sort by cycle
    order = np.argsort(cyc)
    cyc, qd, qc = cyc[order], qd[order], qc[order]
    # Cycle life: first cycle where Qd < 0.8 * Q_initial, else fall back to
    # the published value in CELL_INFO (cycle_life_true).
    q_init = np.nanmax(qd[:5]) if len(qd) >= 5 else np.nanmax(qd)
    info = CELL_INFO.get(cell_name, {"EOL_frac": 0.80, "cycle_life_true": int(cyc[-1])})
    thresh = info["EOL_frac"] * q_init
    bad = np.where(qd < thresh)[0]
    if len(bad):
        cycle_life = int(cyc[bad[0]])
    else:
        cycle_life = info.get("cycle_life_true", int(cyc[-1]))
    return {
        "cell_id": cell_name, "cycle": cyc, "Qd": qd, "Qc": qc,
        "cycle_life": cycle_life, "raw_df": big,
        "col_v": col_v, "col_i": col_i, "col_t": col_t, "col_cy": col_cy,
    }


def build_calce_tensor(cell_dict: dict, n_cycles: int = 100,
                       intra_len: int = 64, features=("voltage", "current")) -> dict:
    """Convert one CALCE cell into the (N, F, L) tensor shape expected by
    BSEEarlyPredictDataset. Returns dict with x, mask, proto, y."""
    df = cell_dict["raw_df"]
    col_v = cell_dict["col_v"]; col_i = cell_dict["col_i"]
    col_cy = cell_dict["col_cy"]; col_t = cell_dict["col_t"]
    cyc_sorted = sorted(cell_dict["cycle"].tolist())[:n_cycles]
    F = len(features)
    x = np.zeros((n_cycles, F, intra_len), dtype=np.float32)
    mask = np.zeros(n_cycles, dtype=np.float32)
    for i, cy in enumerate(cyc_sorted):
        sub = df[df[col_cy] == cy]
        if len(sub) < 4:
            continue
        t = sub[col_t].values if col_t else np.arange(len(sub), dtype=float)
        if (t.max() - t.min()) <= 0:
            continue
        t_norm = (t - t.min()) / (t.max() - t.min())
        grid = np.linspace(0, 1, intra_len)
        for fi, fname in enumerate(features):
            col = col_v if fname == "voltage" else (col_i if fname == "current" else None)
            if col is None:
                continue
            y = sub[col].values.astype(np.float32)
            try:
                x[i, fi, :] = np.interp(grid, t_norm, y).astype(np.float32)
            except Exception:
                pass
        mask[i] = 1.0
    np.nan_to_num(x, copy=False, nan=0.0)
    # CALCE protocol is constant (1C charge / 1C discharge typically)
    proto = np.array([1.0, 80.0, 1.0], dtype=np.float32)
    return dict(x=x, mask=mask, proto=proto,
                y=np.float32(cell_dict["cycle_life"]))


def load_all_calce(root: str, cells: List[str] | None = None,
                   n_cycles: int = 100, intra_len: int = 64) -> list[dict]:
    root = Path(root)
    cells = cells or list(CELL_INFO.keys())
    out = []
    for cn in cells:
        cell_dir = root / cn
        if not cell_dir.exists():
            cell_dir = root / f"{cn}"
            if not cell_dir.exists():
                LOG.warning("calce cell dir not found: %s", cn)
                continue
        cd = load_calce_cell(cell_dir, cn)
        if cd is None:
            continue
        tens = build_calce_tensor(cd, n_cycles=n_cycles, intra_len=intra_len)
        tens["cell_id"] = cn
        tens["chem"] = CELL_INFO.get(cn, {}).get("chem", "?")
        out.append(tens)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/calce")
    args = ap.parse_args()
    cells = load_all_calce(args.root)
    print(f"loaded {len(cells)} CALCE cells")
    for c in cells:
        print(f"  {c['cell_id']}: cycle_life={c['y']}, x.shape={c['x'].shape}, mask sum={c['mask'].sum():.0f}")
