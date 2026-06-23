"""Identify which cells are in fold 0 of our random KFold (seed=42),
and check why all models degrade on this fold."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sklearn.model_selection import KFold

# load manifest
manifest = pd.read_csv("data/interim/mit_meta.csv")
manifest = manifest.dropna(subset=["cycle_life"])
manifest = manifest[manifest["cycle_life"] > 100]
manifest = manifest.sort_values("cell_id").reset_index(drop=True)

# Use the same shuffle as train.py
cells_all = manifest["cell_id"].tolist()
rng = np.random.RandomState(42)
perm = rng.permutation(len(cells_all))
cells_all = [cells_all[i] for i in perm]

kf = KFold(n_splits=5)
for fi, (tr_idx, te_idx) in enumerate(kf.split(cells_all)):
    test_cells = [cells_all[i] for i in te_idx]
    test_df = manifest[manifest["cell_id"].isin(test_cells)]
    print(f"\n=== Fold {fi} test set (n={len(test_df)}) ===")
    print(f"  cycle_life: min={test_df['cycle_life'].min()}, "
          f"med={test_df['cycle_life'].median()}, "
          f"max={test_df['cycle_life'].max()}, "
          f"std={test_df['cycle_life'].std():.0f}")
    print(f"  batch: {test_df['batch'].value_counts().to_dict()}")
    print(f"  policy unique: {test_df['policy_readable'].nunique()}")
    # extreme cells
    bottom = test_df.nsmallest(3, "cycle_life")
    top = test_df.nlargest(3, "cycle_life")
    print(f"  shortest 3: {bottom[['cell_id','cycle_life','policy_readable']].values.tolist()}")
    print(f"  longest 3:  {top[['cell_id','cycle_life','policy_readable']].values.tolist()}")
    # train policy coverage
    train_cells = [cells_all[i] for i in tr_idx]
    train_df = manifest[manifest["cell_id"].isin(train_cells)]
    test_only_policies = set(test_df["policy_readable"]) - set(train_df["policy_readable"])
    print(f"  test-only policies (not in train): {len(test_only_policies)}/"
          f"{test_df['policy_readable'].nunique()}")
