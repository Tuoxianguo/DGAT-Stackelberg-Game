"""
Loader for BSEBench-format Severson 2019 parquet (HuggingFace
bsebench-org/severson-2019).

Each cell is one parquet file `b1c0.parquet`, `b1c1.parquet`, ..., `b3c45.parquet`.
The schema follows BPX-1.1 / BSEBench TimeSeriesSchema. Typical columns:

    cell_id, cycle_index, time_s, voltage_V, current_A, temperature_C,
    discharge_capacity_Ah, charge_capacity_Ah, ... (see HF README)

Plus a metadata row per cell containing `cycle_life`, `policy`, `chemistry`,
etc.  Exact schema is verified at load time and adapted if columns differ.

Usage:
    loader = BSEBenchLoader("data/raw/mit_hf")
    df_manifest = loader.build_manifest()             # one row per cell
    cell = loader.load_cell("b1c0")                   # full time-series
    summary = loader.summarize_cell("b1c0")           # per-cycle aggregates
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..utils.logging_utils import get_logger

LOG = get_logger("bsebench_loader")

# Severson policy regex: e.g. "5.4C(80%)-3.6C"
_POLICY_RE_FULL = re.compile(r"([\d.]+)C\(([\d.]+)%\)-([\d.]+)C")
_POLICY_RE_TWO = re.compile(r"([\d.]+)C-([\d.]+)C")
_POLICY_RE_ONE = re.compile(r"([\d.]+)C")


def parse_policy(policy: str | None) -> dict:
    if policy is None or (isinstance(policy, float) and np.isnan(policy)):
        return {"CC1": np.nan, "SOC_switch": np.nan, "CC2": np.nan, "raw": ""}
    p = str(policy).strip().replace("newstructure", "").strip("-_ ")
    m = _POLICY_RE_FULL.search(p)
    if m:
        return {"CC1": float(m.group(1)), "SOC_switch": float(m.group(2)),
                "CC2": float(m.group(3)), "raw": p}
    m = _POLICY_RE_TWO.search(p)
    if m:
        return {"CC1": float(m.group(1)), "SOC_switch": np.nan,
                "CC2": float(m.group(2)), "raw": p}
    m = _POLICY_RE_ONE.search(p)
    if m:
        return {"CC1": float(m.group(1)), "SOC_switch": np.nan,
                "CC2": np.nan, "raw": p}
    return {"CC1": np.nan, "SOC_switch": np.nan, "CC2": np.nan, "raw": p}


@dataclass
class BSECell:
    cell_id: str
    df: pd.DataFrame           # raw long-form time-series
    cycle_life: int | None
    policy: str | None
    protocol: dict
    chemistry: str | None
    batch: str
    meta: dict


class BSEBenchLoader:
    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(self.root)

    def list_cells(self) -> list[str]:
        return sorted(
            p.stem for p in self.root.glob("*.parquet")
            if re.match(r"b\d+c\d+", p.stem)
        )

    def _file(self, cell_id: str) -> Path:
        return self.root / f"{cell_id}.parquet"

    def load_cell(self, cell_id: str,
                  external_meta: dict | None = None) -> BSECell:
        """Load one cell.

        external_meta : optional dict {cell_id: {cycle_life, policy, ...}} merged
                        from the Tier-1 raw .mat metadata (since BSEBench Tier-2
                        parquet drops these fields).
        """
        df = pd.read_parquet(self._file(cell_id))
        df.columns = [c.lower() for c in df.columns]   # normalize case
        # First try inline metadata (some cells might have it)
        cycle_life = self._extract_scalar(df, ["cycle_life", "rul_cycle", "eol_cycle"])
        policy = self._extract_scalar(df, ["policy", "policy_readable",
                                           "charging_policy"], scalar_str=True)
        chemistry = self._extract_scalar(df, ["chemistry", "cathode"], scalar_str=True)
        # Fallback 1: external metadata (preferred when available)
        if external_meta and cell_id in external_meta:
            em = external_meta[cell_id]
            cycle_life = em.get("cycle_life", cycle_life)
            policy = em.get("policy", policy)
            chemistry = em.get("chemistry", chemistry) or "LFP/graphite"
        # Fallback 2: derive cycle_life from cycle_number (when ground-truth absent)
        if cycle_life is None or (isinstance(cycle_life, float) and pd.isna(cycle_life)):
            if "cycle_number" in df.columns:
                cycle_life = int(df["cycle_number"].max())
            elif "cycle_index" in df.columns:
                cycle_life = int(df["cycle_index"].max())
            elif "cycle" in df.columns:
                cycle_life = int(df["cycle"].max())
        batch = cell_id.split("c")[0]
        protocol = parse_policy(policy)
        return BSECell(
            cell_id=cell_id, df=df, cycle_life=cycle_life, policy=policy,
            protocol=protocol, chemistry=chemistry, batch=batch,
            meta={"n_rows": len(df), "columns": list(df.columns)},
        )

    @staticmethod
    def _extract_scalar(df: pd.DataFrame, candidates: Iterable[str], scalar_str: bool = False):
        for c in candidates:
            if c in df.columns:
                v = df[c].dropna()
                if len(v):
                    val = v.iloc[0]
                    if scalar_str:
                        return None if pd.isna(val) else str(val)
                    return None if pd.isna(val) else int(val)
        return None

    def summarize_cell(self, cell_id: str, external_meta: dict | None = None) -> pd.DataFrame:
        """Per-cycle aggregates (one row per cycle)."""
        cell = self.load_cell(cell_id, external_meta=external_meta)
        df = cell.df
        # Normalize cycle column name
        for cand in ["cycle_number", "cycle_index", "cycle"]:
            if cand in df.columns:
                if cand != "cycle_index":
                    df = df.rename(columns={cand: "cycle_index"})
                break
        else:
            raise KeyError(f"cell {cell_id}: no cycle column")
        # Identify physical column aliases (case already normalized to lower)
        col_v = self._first_col(df, ["voltage_v", "voltage"])
        col_i = self._first_col(df, ["current_a", "current"])
        col_t = self._first_col(df, ["temperature_c", "temp", "t_cell"])
        col_qd = self._first_col(df, ["discharge_capacity_ah", "qd", "q_discharge",
                                      "capacity_ah"])
        col_qc = self._first_col(df, ["charge_capacity_ah", "qc", "q_charge"])
        col_time = self._first_col(df, ["time_s", "t", "elapsed_s"])
        agg = df.groupby("cycle_index", sort=True).agg(
            Tavg=(col_t, "mean") if col_t else (df.columns[0], "size"),
            Tmax=(col_t, "max") if col_t else (df.columns[0], "size"),
            Tmin=(col_t, "min") if col_t else (df.columns[0], "size"),
            Vmax=(col_v, "max") if col_v else (df.columns[0], "size"),
            Vmin=(col_v, "min") if col_v else (df.columns[0], "size"),
            Imax=(col_i, "max") if col_i else (df.columns[0], "size"),
            Imin=(col_i, "min") if col_i else (df.columns[0], "size"),
            Qd_max=(col_qd, "max") if col_qd else (df.columns[0], "size"),
            Qc_max=(col_qc, "max") if col_qc else (df.columns[0], "size"),
            charge_time=(col_time, lambda s: float(s.max() - s.min())) if col_time else (df.columns[0], "size"),
            n_points=(df.columns[0], "size"),
        ).reset_index()
        agg = agg.rename(columns={"cycle_index": "cycle"})
        return agg

    @staticmethod
    def _first_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def build_manifest(self, out_csv: str | os.PathLike | None = None) -> pd.DataFrame:
        rows = []
        for cid in self.list_cells():
            try:
                cell = self.load_cell(cid)
                p = cell.protocol
                rows.append(dict(
                    cell_id=cid, batch=cell.batch, cycle_life=cell.cycle_life,
                    policy=cell.policy, chemistry=cell.chemistry,
                    CC1=p.get("CC1"), SOC_switch=p.get("SOC_switch"),
                    CC2=p.get("CC2"), n_rows=len(cell.df),
                ))
            except Exception as e:  # noqa
                LOG.warning("cell %s failed: %s", cid, e)
        df = pd.DataFrame(rows)
        if out_csv is not None:
            Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv, index=False)
        return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/mit_hf")
    ap.add_argument("--out", default="data/interim/mit_manifest.csv")
    ap.add_argument("--inspect", default=None, help="cell_id to print schema for")
    args = ap.parse_args()
    loader = BSEBenchLoader(args.root)
    if args.inspect:
        cell = loader.load_cell(args.inspect)
        print("cell:", cell.cell_id, "rows:", len(cell.df))
        print("columns:", list(cell.df.columns))
        print(cell.df.head())
        print("cycle_life:", cell.cycle_life, "policy:", cell.policy,
              "chem:", cell.chemistry, "protocol:", cell.protocol)
    else:
        df = loader.build_manifest(out_csv=args.out)
        print(df.head())
        print("N cells:", len(df))
        print("By batch:", df.groupby("batch").size().to_dict())
        print("Cycle life: min=%d, med=%d, max=%d"
              % (df["cycle_life"].min(), df["cycle_life"].median(), df["cycle_life"].max()))
