"""
Torch Dataset / DataLoader for early life prediction.

Takes the BSEBench parquet directory and serves padded tensors of shape:
  x      : (B, N, F, L)        N = first n_cycles cycles, L = intra_len resample
  mask   : (B, N) {0,1}
  proto  : (B, d_proto)        CC1, SOC_switch, CC2
  y      : (B,)                cycle_life
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .bsebench_loader import BSEBenchLoader
from ..utils.logging_utils import get_logger

LOG = get_logger("dataset")


def _interp_to_grid(t: np.ndarray, y: np.ndarray, L: int) -> np.ndarray:
    if len(t) < 2:
        return np.full(L, np.nan, dtype=np.float32)
    t = (t - t.min())
    if t.max() <= 0:
        return np.full(L, np.nan, dtype=np.float32)
    t_norm = t / t.max()
    grid = np.linspace(0, 1, L)
    return np.interp(grid, t_norm, y).astype(np.float32)


class BSEEarlyPredictDataset(Dataset):
    """Pre-extracts early cycle tensor per cell, caches on disk as .npz.

    Default features use the BSEBench schema (lower-cased aliases of
    voltage_V, current_A, temperature_C, capacity_Ah, ...).

    Supports optional cycle-level data augmentation (training only):
        - cycle_dropout : randomly mask 0-20% of cycles per sample
        - gaussian_noise: σ multiplied by per-channel std

    NEW: supports auxiliary 15-d Severson v2 hand-crafted features
    (loaded from a CSV `severson_features.csv`) returned under key `feat_aux`.
    These features are the SAME ones used to fit the Severson Elastic Net
    baseline (8.46% train MAPE), so the deep model can learn an additive
    correction on top of the EN signal.
    """

    def __init__(self, loader: BSEBenchLoader, cell_ids: Sequence[str],
                 n_cycles: int = 100, intra_len: int = 128,
                 features: Sequence[str] = ("voltage_v", "current_a",
                                            "temperature_c", "capacity_ah"),
                 cache_dir: str | Path | None = None,
                 external_meta: dict | None = None,
                 augment: bool = False,
                 cycle_dropout_p: float = 0.10,
                 noise_std: float = 0.01,
                 hybrid_features_csv: str | None = None,
                 hybrid_feature_cols: Sequence[str] | None = None):
        self.loader = loader
        self.cell_ids = list(cell_ids)
        self.n_cycles = n_cycles
        self.intra_len = intra_len
        self.features = list(features)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.external_meta = external_meta or {}
        self.augment = augment
        self.cycle_dropout_p = cycle_dropout_p
        self.noise_std = noise_std
        self._cache: dict = {}

        # Load hybrid auxiliary features keyed by cell_id
        self.hybrid_feat = None
        self.hybrid_dim = 0
        if hybrid_features_csv is not None and Path(hybrid_features_csv).exists():
            import pandas as pd
            hf = pd.read_csv(hybrid_features_csv)
            if hybrid_feature_cols is None:
                # Default: all numeric columns except cell_id/cycle_life/batch/policy
                exclude = {"cell_id", "cycle_life", "batch", "policy"}
                hybrid_feature_cols = [c for c in hf.columns
                                        if c not in exclude
                                        and pd.api.types.is_numeric_dtype(hf[c])]
            self.hybrid_feature_cols = list(hybrid_feature_cols)
            # Z-score normalize each feature (robust to scale)
            from sklearn.preprocessing import StandardScaler
            # fit only on cells in this dataset that exist in the csv (or all of csv)
            sc = StandardScaler()
            X_all = hf[self.hybrid_feature_cols].values
            X_filled = np.nan_to_num(X_all, nan=np.nanmean(X_all))
            sc.fit(X_filled)
            self.hybrid_scaler = sc
            self.hybrid_feat = {}
            for _, r in hf.iterrows():
                vals = r[self.hybrid_feature_cols].values.astype(float)
                vals = np.nan_to_num(vals, nan=np.nanmean(X_all))
                self.hybrid_feat[r["cell_id"]] = sc.transform(vals.reshape(1, -1))[0].astype(np.float32)
            self.hybrid_dim = len(self.hybrid_feature_cols)

    def __len__(self) -> int:
        return len(self.cell_ids)

    def _build_one(self, cid: str) -> dict:
        cache_path = self.cache_dir / f"{cid}.npz" if self.cache_dir else None
        if cache_path is not None and cache_path.exists():
            d = np.load(cache_path)
            return {k: torch.from_numpy(d[k]) for k in d.files}
        cell = self.loader.load_cell(cid, external_meta=self.external_meta)
        df = cell.df
        # Normalize cycle column name
        for cand in ["cycle_number", "cycle_index", "cycle"]:
            if cand in df.columns:
                if cand != "cycle_index":
                    df = df.rename(columns={cand: "cycle_index"})
                break
        # Choose first n_cycles cycle indices that have ≥4 rows
        counts = df.groupby("cycle_index").size()
        valid_cycles = counts[counts >= 4].index.tolist()
        valid_cycles = sorted(valid_cycles)[: self.n_cycles]
        F = len(self.features)
        x = np.zeros((self.n_cycles, F, self.intra_len), dtype=np.float32)
        mask = np.zeros(self.n_cycles, dtype=np.float32)
        time_col = "time_s" if "time_s" in df.columns else ("t" if "t" in df.columns else None)
        for i, cy in enumerate(valid_cycles):
            sub = df[df["cycle_index"] == cy]
            if len(sub) < 4:
                continue
            t = sub[time_col].values if time_col else np.arange(len(sub), dtype=float)
            for fi, fname in enumerate(self.features):
                if fname not in sub.columns:
                    continue
                y = sub[fname].values.astype(np.float32)
                x[i, fi, :] = _interp_to_grid(t, y, self.intra_len)
            mask[i] = 1.0
        np.nan_to_num(x, copy=False, nan=0.0)
        proto = cell.protocol
        # Default values for cells without parsed protocol (BSEBench Tier-2
        # parquet drops policy strings). We use mid-range C-rates so the
        # GNN graph builder still has valid nodes.
        cc1_v = proto.get("CC1")
        s_v = proto.get("SOC_switch")
        cc2_v = proto.get("CC2")
        if cc1_v is None or (isinstance(cc1_v, float) and np.isnan(cc1_v)):
            cc1_v = 4.0
        if s_v is None or (isinstance(s_v, float) and np.isnan(s_v)):
            s_v = 60.0
        if cc2_v is None or (isinstance(cc2_v, float) and np.isnan(cc2_v)):
            cc2_v = 4.0
        p = np.array([cc1_v, s_v, cc2_v], dtype=np.float32)
        y = float(cell.cycle_life or 0)
        rec = dict(x=x, mask=mask, proto=p, y=np.float32(y))
        # Attach hybrid features if available (zero vector otherwise)
        if self.hybrid_feat is not None:
            feat = self.hybrid_feat.get(cid, np.zeros(self.hybrid_dim, dtype=np.float32))
            rec["feat_aux"] = feat.astype(np.float32)
        if cache_path is not None:
            np.savez_compressed(cache_path, **rec)
        return {k: torch.from_numpy(np.asarray(v)) if not isinstance(v, np.ndarray)
                else torch.from_numpy(v) for k, v in rec.items()}

    def __getitem__(self, idx: int) -> dict:
        cid = self.cell_ids[idx]
        if cid in self._cache:
            out = self._cache[cid]
        else:
            out = self._build_one(cid)
            self._cache[cid] = out
        if not self.augment:
            return out
        # Augmentation: cycle dropout + tiny Gaussian noise
        x = out["x"].clone()
        m = out["mask"].clone()
        if self.cycle_dropout_p > 0:
            n_valid = int(m.sum().item())
            n_drop = int(n_valid * self.cycle_dropout_p)
            if n_drop > 0:
                idx_valid = torch.nonzero(m > 0).squeeze(-1)
                drop_pos = idx_valid[torch.randperm(len(idx_valid))[:n_drop]]
                m[drop_pos] = 0.0
                x[drop_pos] = 0.0
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std * x.std()
        ret = {"x": x, "mask": m, "proto": out["proto"], "y": out["y"]}
        if "feat_aux" in out:
            ret["feat_aux"] = out["feat_aux"]
        return ret


def severson_split(manifest_df: pd.DataFrame,
                   batch_col: str = "batch") -> dict[str, list[str]]:
    """Severson 2019 official split: Batch1=train, Batch2=primary test, Batch3=secondary test."""
    train = manifest_df[manifest_df[batch_col] == "b1"]["cell_id"].tolist()
    test1 = manifest_df[manifest_df[batch_col] == "b2"]["cell_id"].tolist()
    test2 = manifest_df[manifest_df[batch_col] == "b3"]["cell_id"].tolist()
    return {"train": train, "test_primary": test1, "test_secondary": test2}
