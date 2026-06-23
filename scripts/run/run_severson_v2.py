"""
Severson Elastic Net baseline v2 using OFFICIAL Severson features
(ΔQ_{100-10} variance + Qd / IR / Tavg / chargetime), reproducing
the Nature Energy 2019 setup.

Compares both:
  - 5-fold random CV
  - Severson official batch split (B1 train, B2 primary test, B3 secondary)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from battery_paper.features import compute_severson_features_v2
from battery_paper.models.baselines import train_severson_elastic_net
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _eval_batch_split(features_df: pd.DataFrame, log_target: bool = True) -> dict:
    """Severson official: train on B1, test_primary on B2, test_secondary on B3."""
    feats = ["log_var_dq", "log_abs_min_dq", "skew_dq", "kurt_dq", "log_mean_dq",
             "slope_2_100", "intercept_2_100", "qd_2", "qd_100", "qd_diff_100_2",
             "IR_min_2_100", "IR_at_100_minus_2", "Tavg_integral_2_100",
             "Tavg_max_2_100", "chgt_first5"]
    df = features_df.dropna(subset=feats + ["cycle_life"]).copy()
    df = df[df["cycle_life"] > 0]
    feats = [f for f in feats if f in df.columns]
    X = df[feats].values.astype(float)
    y = df["cycle_life"].values.astype(float)
    y_lab = np.log10(y) if log_target else y
    train_mask = df["batch"] == "b1"
    test1_mask = df["batch"] == "b2"
    test2_mask = df["batch"] == "b3"
    pipe = Pipeline([("sc", StandardScaler()),
                     ("en", ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                                          alphas=np.logspace(-4, 2, 50), cv=5,
                                          max_iter=30000, random_state=42))])
    pipe.fit(X[train_mask], y_lab[train_mask])

    def _metric(mask):
        if mask.sum() == 0:
            return {"MAPE": float("nan"), "RMSE": float("nan"), "n": 0}
        pred = pipe.predict(X[mask])
        pred_cy = np.power(10.0, pred) if log_target else pred
        true_cy = np.power(10.0, y_lab[mask]) if log_target else y_lab[mask]
        return {"MAPE": float(mean_absolute_percentage_error(true_cy, pred_cy)) * 100,
                "RMSE": float(np.sqrt(mean_squared_error(true_cy, pred_cy))),
                "n": int(mask.sum()),
                "alpha": float(pipe.named_steps["en"].alpha_),
                "l1_ratio": float(pipe.named_steps["en"].l1_ratio_)}

    return {"train": _metric(train_mask), "test_primary": _metric(test1_mask),
            "test_secondary": _metric(test2_mask), "n_features": len(feats),
            "features": feats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="data/interim/mit_summary.parquet")
    ap.add_argument("--out_dir", default="experiments/run_severson_v2")
    args = ap.parse_args()

    df = pd.read_parquet(args.summary)
    print(f"loaded {len(df)} cells from {args.summary}")
    feats = compute_severson_features_v2(df)
    print(f"feats columns: {list(feats.columns)}")
    print(feats.describe().T)
    feats["cell_id"] = df["cell_id"].values
    feats["cycle_life"] = df["cycle_life"].values
    feats["batch"] = df["batch"].values
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    feats.to_csv(Path(args.out_dir) / "features_v2.csv", index=False)

    print("\n=== Severson batch split ===")
    res = _eval_batch_split(feats)
    print(json.dumps({k: v for k, v in res.items() if k != "features"},
                     indent=2, default=str))

    print("\n=== 5-fold CV ===")
    bundle = train_severson_elastic_net(feats, feature_names=res["features"])
    cv_summary = {"cv": bundle.cv_metrics, "batch_split": res}
    with open(Path(args.out_dir) / "results.json", "w") as f:
        json.dump(cv_summary, f, indent=2, default=str)
    print("\nDone. Saved to", args.out_dir)


if __name__ == "__main__":
    main()
