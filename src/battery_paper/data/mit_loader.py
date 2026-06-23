"""
MIT / Stanford / Toyota dataset loader (Severson et al., Nature Energy 2019).

The official files are HDF5-based MATLAB v7.3 .mat files with deeply nested
struct-of-cells. We load them with `h5py`, flatten one cell at a time, and
emit a tidy parquet store keyed by (batch, cell_id).

Per-cell output schema (parquet):
  - summary.parquet : one row per cycle (cycle, Qd, Qc, IR, Tavg, Tmax, Tmin, chargetime)
  - cycles/         : one parquet per cycle with columns (t, V, I, T, Q, Qd, Qc, dQdV)
  - protocol.json   : charging policy parameters (CC1, Q1, CC2, V_cutoff, ...)
  - meta.json       : cycle_life, barcode, channel_id

Usage:
    loader = MITLoader(raw_dir="data/raw/mit",
                       out_dir="data/interim/mit")
    loader.preprocess_all()        # one-time
    cells = loader.list_cells()
    cell  = loader.load_cell("b1c0")
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import h5py
import numpy as np
import pandas as pd

from ..utils.logging_utils import get_logger

LOG = get_logger("mit_loader")

# 3 official batches; key = short name, value = filename suffix
BATCH_FILES = {
    "b1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "b2": "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
    "b3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}

# Cells that are known-bad in Severson's notebook (low cycle count, sensor faults).
# We keep them but flag in metadata.
KNOWN_BAD = {
    "b1": [0, 1, 3, 8, 10, 12, 13, 22],
    "b2": [7, 8, 9, 15, 16],
    "b3": [37, 2, 23, 32, 38, 39, 40, 41, 42, 43, 44, 45],
}


@dataclass
class MITCellSummary:
    """Summary row per cycle (vector of length n_cycles)."""

    cycle: np.ndarray
    Qd: np.ndarray
    Qc: np.ndarray
    IR: np.ndarray
    Tavg: np.ndarray
    Tmax: np.ndarray
    Tmin: np.ndarray
    chargetime: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "cycle": self.cycle,
                "Qd": self.Qd,
                "Qc": self.Qc,
                "IR": self.IR,
                "Tavg": self.Tavg,
                "Tmax": self.Tmax,
                "Tmin": self.Tmin,
                "chargetime": self.chargetime,
            }
        )


@dataclass
class MITCell:
    batch: str
    cell_id: str
    barcode: str
    channel_id: int
    cycle_life: int
    policy: str
    protocol: dict
    summary: MITCellSummary
    cycles: dict[int, pd.DataFrame] = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Protocol parsing
# ----------------------------------------------------------------------------
def parse_policy_string(policy: str) -> dict:
    """Parse Severson policy strings like '5.4C(80%)-3.6C' or '4.4C(40%)-6C-newstructure'.

    Returns a dict with CC1, SOC_switch, CC2, V_cutoff, special flags.
    """
    p = policy.strip()
    out = {
        "policy_raw": p,
        "CC1": np.nan,
        "SOC_switch": np.nan,
        "CC2": np.nan,
        "V_cutoff": 3.6,  # MIT default upper cutoff in 1C nominal
        "is_newstructure": "newstructure" in p,
    }
    p = p.replace("newstructure", "").strip("-_ ")
    # Pattern: "<CC1>C(<SOC>%)-<CC2>C"
    m = re.match(r"^([\d.]+)C\(([\d.]+)%\)-([\d.]+)C$", p)
    if m:
        out["CC1"] = float(m.group(1))
        out["SOC_switch"] = float(m.group(2))
        out["CC2"] = float(m.group(3))
        return out
    # Pattern: "<CC1>C-<CC2>C" (no SOC switch)
    m = re.match(r"^([\d.]+)C-([\d.]+)C$", p)
    if m:
        out["CC1"] = float(m.group(1))
        out["CC2"] = float(m.group(2))
        return out
    # Pattern: "<CC>C" single-rate
    m = re.match(r"^([\d.]+)C$", p)
    if m:
        out["CC1"] = float(m.group(1))
        return out
    return out


# ----------------------------------------------------------------------------
# Low-level h5py walker
# ----------------------------------------------------------------------------
def _deref(f: h5py.File, ref) -> np.ndarray:
    """Dereference an HDF5 object ref (returns numpy array)."""
    return np.asarray(f[ref])


def _to_str(f: h5py.File, ref) -> str:
    arr = _deref(f, ref)
    return "".join(chr(int(c)) for c in arr.flatten())


def _to_scalar(f: h5py.File, ref) -> float:
    arr = _deref(f, ref).flatten()
    return float(arr[0]) if len(arr) else float("nan")


def _read_batch(mat_path: str, batch_key: str) -> Iterator[MITCell]:
    """Yield MITCell objects from one MIT .mat batch file."""
    with h5py.File(mat_path, "r") as f:
        # The top-level struct: f['batch']
        if "batch" not in f:
            raise KeyError(f"'batch' group not found in {mat_path}; keys={list(f.keys())}")
        batch = f["batch"]
        n_cells = batch["cycle_life"].shape[0]
        LOG.info("opened %s, %d cells in batch %s", mat_path, n_cells, batch_key)
        for i in range(n_cells):
            try:
                cl_ref = batch["cycle_life"][i, 0]
                cycle_life = int(_to_scalar(f, cl_ref))
                barcode_ref = batch["barcode"][i, 0]
                barcode = _to_str(f, barcode_ref)
                channel_ref = batch["channel_id"][i, 0]
                channel_id = int(_to_scalar(f, channel_ref))
                policy_ref = batch["policy_readable"][i, 0]
                policy = _to_str(f, policy_ref)
                # ---- summary ----
                summ_ref = batch["summary"][i, 0]
                summ = f[summ_ref]
                cycle = _deref(f, summ["cycle"][0, 0]).squeeze().astype(int)
                Qd = _deref(f, summ["QD"][0, 0]).squeeze()
                Qc = _deref(f, summ["QC"][0, 0]).squeeze()
                IR = _deref(f, summ["IR"][0, 0]).squeeze()
                Tavg = _deref(f, summ["Tavg"][0, 0]).squeeze()
                Tmax = _deref(f, summ["Tmax"][0, 0]).squeeze()
                Tmin = _deref(f, summ["Tmin"][0, 0]).squeeze()
                chargetime = _deref(f, summ["chargetime"][0, 0]).squeeze()
                summary = MITCellSummary(
                    cycle=np.atleast_1d(cycle),
                    Qd=np.atleast_1d(Qd),
                    Qc=np.atleast_1d(Qc),
                    IR=np.atleast_1d(IR),
                    Tavg=np.atleast_1d(Tavg),
                    Tmax=np.atleast_1d(Tmax),
                    Tmin=np.atleast_1d(Tmin),
                    chargetime=np.atleast_1d(chargetime),
                )
                # ---- per-cycle measurements ----
                cycles = {}
                cy_ref = batch["cycles"][i, 0]
                cy = f[cy_ref]
                n_cy = cy["I"].shape[0]
                # cycle 0 is often baseline, keep all
                for k in range(n_cy):
                    try:
                        I = _deref(f, cy["I"][k, 0]).squeeze()
                        V = _deref(f, cy["V"][k, 0]).squeeze()
                        T = _deref(f, cy["T"][k, 0]).squeeze()
                        Qc_k = _deref(f, cy["Qc"][k, 0]).squeeze()
                        Qd_k = _deref(f, cy["Qd"][k, 0]).squeeze()
                        t = _deref(f, cy["t"][k, 0]).squeeze()
                        # discharge dQdV is provided on a fixed V grid (1000)
                        # but here we keep raw (t, V, I, ...)
                        df = pd.DataFrame(
                            {
                                "t": np.atleast_1d(t),
                                "V": np.atleast_1d(V),
                                "I": np.atleast_1d(I),
                                "T": np.atleast_1d(T),
                                "Qc": np.atleast_1d(Qc_k),
                                "Qd": np.atleast_1d(Qd_k),
                            }
                        )
                        cycles[k] = df
                    except Exception as e:  # noqa
                        LOG.debug("batch %s cell %d cycle %d skipped: %s", batch_key, i, k, e)
                cell_id = f"{batch_key}c{i}"
                protocol = parse_policy_string(policy)
                yield MITCell(
                    batch=batch_key,
                    cell_id=cell_id,
                    barcode=barcode,
                    channel_id=channel_id,
                    cycle_life=cycle_life,
                    policy=policy,
                    protocol=protocol,
                    summary=summary,
                    cycles=cycles,
                )
            except Exception as e:  # noqa
                LOG.exception("failed to load batch %s cell %d: %s", batch_key, i, e)


# ----------------------------------------------------------------------------
# Public loader
# ----------------------------------------------------------------------------
class MITLoader:
    def __init__(
        self,
        raw_dir: str | os.PathLike,
        out_dir: str | os.PathLike,
        n_cycles_keep: int | None = None,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.out_dir = Path(out_dir)
        self.n_cycles_keep = n_cycles_keep
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def preprocess_all(self, overwrite: bool = False) -> None:
        manifest = []
        for bkey, fname in BATCH_FILES.items():
            mat_path = self.raw_dir / fname
            if not mat_path.exists():
                LOG.warning("missing %s; skip", mat_path)
                continue
            for cell in _read_batch(str(mat_path), bkey):
                cdir = self.out_dir / cell.cell_id
                if cdir.exists() and not overwrite:
                    LOG.info("skip %s (exists)", cell.cell_id)
                    manifest.append(self._brief(cell))
                    continue
                if cdir.exists():
                    shutil.rmtree(cdir)
                cdir.mkdir(parents=True)
                # summary
                cell.summary.to_dataframe().to_parquet(cdir / "summary.parquet")
                # per-cycle
                cyc_dir = cdir / "cycles"
                cyc_dir.mkdir()
                keys = sorted(cell.cycles.keys())
                if self.n_cycles_keep is not None:
                    keys = keys[: self.n_cycles_keep]
                for k in keys:
                    cell.cycles[k].to_parquet(cyc_dir / f"cycle_{k:04d}.parquet")
                # metadata
                meta = {
                    "batch": cell.batch,
                    "cell_id": cell.cell_id,
                    "barcode": cell.barcode,
                    "channel_id": cell.channel_id,
                    "cycle_life": cell.cycle_life,
                    "policy": cell.policy,
                    "is_known_bad": (
                        int(cell.cell_id.split("c")[1]) in KNOWN_BAD.get(cell.batch, [])
                    ),
                }
                (cdir / "meta.json").write_text(json.dumps(meta, indent=2))
                (cdir / "protocol.json").write_text(json.dumps(cell.protocol, indent=2))
                manifest.append(self._brief(cell))
                LOG.info("wrote %s (cycle_life=%d, n_cycles=%d, policy=%s)",
                         cell.cell_id, cell.cycle_life, len(cell.cycles), cell.policy)
        # global manifest
        man_df = pd.DataFrame(manifest)
        man_df.to_csv(self.out_dir / "manifest.csv", index=False)
        LOG.info("manifest with %d cells -> %s", len(manifest), self.out_dir / "manifest.csv")

    @staticmethod
    def _brief(cell: MITCell) -> dict:
        return {
            "cell_id": cell.cell_id,
            "batch": cell.batch,
            "cycle_life": cell.cycle_life,
            "policy": cell.policy,
            "CC1": cell.protocol.get("CC1"),
            "SOC_switch": cell.protocol.get("SOC_switch"),
            "CC2": cell.protocol.get("CC2"),
            "n_cycles_data": len(cell.cycles),
        }

    def list_cells(self) -> list[str]:
        return sorted([p.name for p in self.out_dir.iterdir() if p.is_dir()])

    def load_cell(self, cell_id: str, load_cycles: bool = True) -> MITCell:
        cdir = self.out_dir / cell_id
        meta = json.loads((cdir / "meta.json").read_text())
        proto = json.loads((cdir / "protocol.json").read_text())
        s = pd.read_parquet(cdir / "summary.parquet")
        summary = MITCellSummary(
            cycle=s["cycle"].values, Qd=s["Qd"].values, Qc=s["Qc"].values,
            IR=s["IR"].values, Tavg=s["Tavg"].values, Tmax=s["Tmax"].values,
            Tmin=s["Tmin"].values, chargetime=s["chargetime"].values,
        )
        cycles = {}
        if load_cycles:
            for p in sorted((cdir / "cycles").glob("cycle_*.parquet")):
                k = int(p.stem.split("_")[1])
                cycles[k] = pd.read_parquet(p)
        return MITCell(
            batch=meta["batch"], cell_id=meta["cell_id"], barcode=meta["barcode"],
            channel_id=meta["channel_id"], cycle_life=meta["cycle_life"],
            policy=meta["policy"], protocol=proto, summary=summary, cycles=cycles,
        )


def load_mit_processed(out_dir: str | os.PathLike) -> tuple[pd.DataFrame, list[str]]:
    """Quick helper: load manifest df + cell_id list."""
    out_dir = Path(out_dir)
    df = pd.read_csv(out_dir / "manifest.csv")
    return df, df["cell_id"].tolist()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/mit")
    ap.add_argument("--out", default="data/interim/mit")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--n_cycles_keep", type=int, default=None,
                    help="optional cap on per-cell cycle count (debug)")
    args = ap.parse_args()
    MITLoader(args.raw, args.out, n_cycles_keep=args.n_cycles_keep).preprocess_all(
        overwrite=args.overwrite
    )
