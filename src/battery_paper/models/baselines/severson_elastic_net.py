"""
Severson Elastic Net regression baseline (Nature Energy 2019, Table 1).

Pipeline:
  - Take dict-of-features per cell (output of severson_features)
  - log10(cycle_life) regression with ElasticNet (alpha CV, l1_ratio CV)
  - 5-fold CV on the Severson training set (Batch 1)
  - Report MAPE, RMSE on cycle_life scale

Usage:
    from battery_paper.models.baselines import train_severson_elastic_net
    bundle = train_severson_elastic_net(features_df)
    pred = bundle.model.predict(features_df_test[bundle.feature_names])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ...utils.logging_utils import get_logger

LOG = get_logger("baseline.severson")


@dataclass
class SeversonElasticNet:
    model: Pipeline
    feature_names: list[str]
    cv_metrics: dict
    log_target: bool = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        log_pred = self.model.predict(X[self.feature_names].values)
        return np.power(10.0, log_pred) if self.log_target else log_pred


_DEFAULT_FEATS = [
    "log_var_dq", "log_abs_min_dq", "skew_dq", "kurt_dq",
    "slope_2_100", "intercept_2_100",
    "qd_2", "qd_100", "qd_init", "qd_max_min_early",
    "t_avg_2_100", "chgt_first5",
]


def train_severson_elastic_net(
    features_df: pd.DataFrame,
    feature_names: Iterable[str] | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    log_target: bool = True,
) -> SeversonElasticNet:
    feats = list(feature_names) if feature_names else [
        c for c in _DEFAULT_FEATS if c in features_df.columns
    ]
    df = features_df.dropna(subset=feats + ["cycle_life"]).copy()
    df = df[df["cycle_life"] > 0]
    LOG.info("train on %d cells, %d features: %s", len(df), len(feats), feats)
    if len(df) < 20:
        raise ValueError(f"too few cells ({len(df)}) after dropna")

    X = df[feats].values.astype(float)
    y = df["cycle_life"].values.astype(float)
    y_lab = np.log10(y) if log_target else y

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_mape, cv_rmse = [], []
    for fold, (idx_tr, idx_va) in enumerate(kf.split(X)):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("enet", ElasticNetCV(
                l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                alphas=np.logspace(-4, 2, 50),
                cv=5, max_iter=20000, random_state=random_state,
            )),
        ])
        pipe.fit(X[idx_tr], y_lab[idx_tr])
        pred = pipe.predict(X[idx_va])
        pred_cy = np.power(10.0, pred) if log_target else pred
        true_cy = np.power(10.0, y_lab[idx_va]) if log_target else y_lab[idx_va]
        mape = float(mean_absolute_percentage_error(true_cy, pred_cy)) * 100
        rmse = float(np.sqrt(mean_squared_error(true_cy, pred_cy)))
        LOG.info("fold %d: MAPE=%.2f%%, RMSE=%.1f cycles", fold, mape, rmse)
        cv_mape.append(mape); cv_rmse.append(rmse)

    # Final fit on full set
    final = Pipeline([
        ("scaler", StandardScaler()),
        ("enet", ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            alphas=np.logspace(-4, 2, 50), cv=5,
            max_iter=20000, random_state=random_state,
        )),
    ])
    final.fit(X, y_lab)

    metrics = {
        "cv_mape_mean": float(np.mean(cv_mape)),
        "cv_mape_std": float(np.std(cv_mape)),
        "cv_rmse_mean": float(np.mean(cv_rmse)),
        "cv_rmse_std": float(np.std(cv_rmse)),
        "alpha": float(final.named_steps["enet"].alpha_),
        "l1_ratio": float(final.named_steps["enet"].l1_ratio_),
    }
    LOG.info("CV MAPE %.2f±%.2f%%, RMSE %.1f±%.1f, alpha=%.4f, l1_ratio=%.2f",
             metrics["cv_mape_mean"], metrics["cv_mape_std"],
             metrics["cv_rmse_mean"], metrics["cv_rmse_std"],
             metrics["alpha"], metrics["l1_ratio"])
    return SeversonElasticNet(model=final, feature_names=feats,
                              cv_metrics=metrics, log_target=log_target)
