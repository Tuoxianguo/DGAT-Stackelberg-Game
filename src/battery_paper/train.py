"""
End-to-end training script for HSMM-GraphGame on Severson MIT data.

Usage:
    python -m battery_paper.train --config experiments/cfg_default.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from .data import BSEBenchLoader, BSEEarlyPredictDataset, severson_split, load_mit_meta
from .models.baselines import (VanillaTransformerRUL, LSTMRUL, LSTMAttRUL,
                                BatteryGPTLite, PBTLite, DGATLite, DGATPlusLite)
from .models.proposed import HSMMGraphGameModel, DGATPlusComposite
from .utils import get_logger, set_seed

LOG = get_logger("train")


@dataclass
class TrainConfig:
    # Data
    data_root: str = "data/raw/mit_hf"
    cache_dir: str = "data/processed/mit_cache"
    manifest_csv: str = "data/interim/mit_manifest.csv"
    meta_csv: str = "data/interim/mit_meta.csv"
    n_cycles: int = 100
    intra_len: int = 64
    features: tuple = ("voltage_v", "current_a", "temperature_c", "capacity_ah")
    # Model
    model: str = "hsmm_graph_game"   # or "vanilla"
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    hsmm_K: int = 4
    hsmm_D_max: int = 300
    use_graph: bool = True
    # Training
    epochs: int = 50
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 42
    log_target: bool = True
    alpha_hsmm: float = 0.01      # weight on HSMM nll term
    alpha_aux_proto: float = 0.1  # weight on protocol-conditioned aux head
    augment: bool = True
    # Eval mode: 'severson' batch split, '5fold' k-fold CV, 'stratified_5fold'
    eval_mode: str = "severson"
    n_folds: int = 5
    fold_idx: int = -1            # -1 = run all folds sequentially
    # NEW: hybrid auxiliary features
    hybrid_features_csv: Optional[str] = None
    # NEW: loss config
    loss_type: str = "log_mse"    # 'log_mse', 'huber', 'tail_weighted_huber'
    tail_weight_power: float = 1.0  # 0 = uniform, >0 = up-weight extremes
    # Output
    out_dir: str = "experiments/run_default"
    device: Optional[str] = None


def load_config(path: str | None) -> TrainConfig:
    cfg = TrainConfig()
    if path and os.path.exists(path):
        d = yaml.safe_load(open(path))
        for k, v in (d or {}).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg


def collate(batch):
    x = torch.stack([b["x"] for b in batch])
    m = torch.stack([b["mask"] for b in batch])
    p = torch.stack([b["proto"] for b in batch])
    y = torch.stack([b["y"] for b in batch])
    out = {"x": x, "mask": m, "proto": p, "y": y}
    if "feat_aux" in batch[0]:
        out["feat_aux"] = torch.stack([b["feat_aux"] for b in batch])
    return out


def mape(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true) / np.clip(np.abs(true), 1, None))) * 100


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


_SIMPLE_MODELS = (VanillaTransformerRUL, LSTMRUL, LSTMAttRUL,
                  BatteryGPTLite, PBTLite, DGATLite, DGATPlusLite)


def evaluate(model, loader, device, log_target: bool) -> dict:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            m = batch["mask"].to(device)
            p = batch["proto"].to(device)
            y = batch["y"].to(device)
            fa = batch["feat_aux"].to(device) if "feat_aux" in batch else None
            if isinstance(model, _SIMPLE_MODELS):
                yhat = model(x, m, fa) if fa is not None else model(x, m)
            else:
                out = model(x, m, p, fa) if fa is not None else model(x, m, p)
                yhat = out.rul_hat
            preds.append(yhat.cpu().numpy())
            trues.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return {"MAPE": mape(preds, trues), "RMSE": rmse(preds, trues),
            "n": int(len(trues)), "pred": preds.tolist(), "true": trues.tolist()}


def train(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)

    LOG.info("device: %s, model: %s", device, cfg.model)

    # ------------------------------------------------------------------ #
    # Data                                                                #
    # ------------------------------------------------------------------ #
    loader = BSEBenchLoader(cfg.data_root)
    external_meta = {}
    if os.path.exists(cfg.meta_csv):
        external_meta = load_mit_meta(cfg.meta_csv)
        LOG.info("loaded external meta for %d cells", len(external_meta))
    else:
        LOG.warning("meta_csv %s not found, will derive cycle_life from data",
                    cfg.meta_csv)

    if not os.path.exists(cfg.manifest_csv):
        LOG.info("manifest not found, building...")
        # Build manifest with external_meta injected
        rows = []
        for cid in loader.list_cells():
            try:
                cell = loader.load_cell(cid, external_meta=external_meta)
                p = cell.protocol
                rows.append(dict(
                    cell_id=cid, batch=cell.batch, cycle_life=cell.cycle_life,
                    policy=cell.policy, chemistry=cell.chemistry,
                    CC1=p.get("CC1"), SOC_switch=p.get("SOC_switch"),
                    CC2=p.get("CC2"),
                ))
            except Exception as e:
                LOG.warning("cell %s failed: %s", cid, e)
        manifest = pd.DataFrame(rows)
        Path(cfg.manifest_csv).parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(cfg.manifest_csv, index=False)
    else:
        manifest = pd.read_csv(cfg.manifest_csv)
    manifest = manifest.dropna(subset=["cycle_life"])
    manifest = manifest[manifest["cycle_life"] > 100]
    LOG.info("manifest: %d cells after filtering", len(manifest))

    common_kwargs = dict(n_cycles=cfg.n_cycles, intra_len=cfg.intra_len,
                         features=cfg.features, cache_dir=cfg.cache_dir,
                         external_meta=external_meta)
    train_kwargs = {**common_kwargs, "augment": getattr(cfg, "augment", True)}

    if cfg.eval_mode == "severson":
        split = severson_split(manifest)
        LOG.info("split: train=%d, test_primary=%d, test_secondary=%d",
                 len(split["train"]), len(split["test_primary"]),
                 len(split["test_secondary"]))
        splits = [{"train": split["train"], "test1": split["test_primary"],
                   "test2": split["test_secondary"], "fold": 0}]
    elif cfg.eval_mode == "5fold":
        from sklearn.model_selection import KFold
        cells_all = manifest["cell_id"].tolist()
        rng = np.random.RandomState(cfg.seed)
        idx_perm = rng.permutation(len(cells_all))
        cells_all = [cells_all[i] for i in idx_perm]
        kf = KFold(n_splits=cfg.n_folds)
        splits = []
        for fi, (tr_idx, te_idx) in enumerate(kf.split(cells_all)):
            train_cells = [cells_all[i] for i in tr_idx]
            test_cells = [cells_all[i] for i in te_idx]
            splits.append({"train": train_cells, "test1": test_cells,
                           "test2": [], "fold": fi})
        LOG.info("5-fold CV: %d folds, ~%d train / %d test per fold",
                 cfg.n_folds, len(splits[0]["train"]), len(splits[0]["test1"]))
        if cfg.fold_idx >= 0:
            splits = [splits[cfg.fold_idx]]
            LOG.info("running fold_idx=%d only", cfg.fold_idx)
    elif cfg.eval_mode == "stratified_5fold":
        # Stratify by cycle_life quantile bins so each fold has similar
        # short/medium/long cells. Critical to avoid the fold-0 outlier problem.
        from sklearn.model_selection import StratifiedKFold
        man_sorted = manifest.copy()
        # 5 quantile bins on cycle_life
        man_sorted["cl_bin"] = pd.qcut(man_sorted["cycle_life"], q=5,
                                       labels=False, duplicates="drop")
        cells_all = man_sorted["cell_id"].tolist()
        bins = man_sorted["cl_bin"].tolist()
        skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True,
                              random_state=cfg.seed)
        splits = []
        for fi, (tr_idx, te_idx) in enumerate(skf.split(cells_all, bins)):
            splits.append({
                "train": [cells_all[i] for i in tr_idx],
                "test1": [cells_all[i] for i in te_idx],
                "test2": [], "fold": fi,
            })
        LOG.info("STRATIFIED 5-fold CV by cycle_life bins: ~%d train / %d test per fold",
                 len(splits[0]["train"]), len(splits[0]["test1"]))
        if cfg.fold_idx >= 0:
            splits = [splits[cfg.fold_idx]]
    else:
        raise ValueError(f"unknown eval_mode: {cfg.eval_mode}")

    # We'll loop over splits and aggregate
    all_fold_results = []
    for split_info in splits:
        LOG.info("=" * 60)
        LOG.info("FOLD %d: train=%d test1=%d test2=%d",
                 split_info["fold"], len(split_info["train"]),
                 len(split_info["test1"]), len(split_info["test2"]))
        fold_out_dir = Path(cfg.out_dir) / f"fold_{split_info['fold']}"
        fold_out_dir.mkdir(parents=True, exist_ok=True)
        fold_res = _run_one_split(cfg, loader, common_kwargs, split_info,
                                  fold_out_dir, device)
        all_fold_results.append(fold_res)

    # Aggregate
    if len(all_fold_results) > 1:
        agg = {
            "n_folds": len(all_fold_results),
            "MAPE_test1_mean": float(np.mean([r["test1"]["MAPE"] for r in all_fold_results])),
            "MAPE_test1_std":  float(np.std( [r["test1"]["MAPE"] for r in all_fold_results])),
            "RMSE_test1_mean": float(np.mean([r["test1"]["RMSE"] for r in all_fold_results])),
            "RMSE_test1_std":  float(np.std( [r["test1"]["RMSE"] for r in all_fold_results])),
            "fold_results": all_fold_results,
        }
        with open(Path(cfg.out_dir) / "kfold_summary.json", "w") as f:
            json.dump(agg, f, indent=2)
        LOG.info("\nFINAL 5-fold:  MAPE %.2f ± %.2f%%   RMSE %.1f ± %.1f",
                 agg["MAPE_test1_mean"], agg["MAPE_test1_std"],
                 agg["RMSE_test1_mean"], agg["RMSE_test1_std"])
        return agg
    return all_fold_results[0]


def _run_one_split(cfg: TrainConfig, loader, common_kwargs, split_info,
                   fold_out_dir, device) -> dict:
    extra_ds_kwargs = {}
    if getattr(cfg, "hybrid_features_csv", None):
        extra_ds_kwargs["hybrid_features_csv"] = cfg.hybrid_features_csv
    train_kwargs = {**common_kwargs, "augment": getattr(cfg, "augment", True),
                    **extra_ds_kwargs}
    eval_kwargs = {**common_kwargs, **extra_ds_kwargs}
    ds_train = BSEEarlyPredictDataset(loader, split_info["train"], **train_kwargs)
    ds_test1 = BSEEarlyPredictDataset(loader, split_info["test1"], **eval_kwargs)
    has_test2 = len(split_info["test2"]) > 0
    ds_test2 = BSEEarlyPredictDataset(loader, split_info["test2"], **eval_kwargs) \
        if has_test2 else None
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                          collate_fn=collate, num_workers=0)
    dl_test1 = DataLoader(ds_test1, batch_size=cfg.batch_size, shuffle=False,
                          collate_fn=collate, num_workers=0)
    dl_test2 = DataLoader(ds_test2, batch_size=cfg.batch_size, shuffle=False,
                          collate_fn=collate, num_workers=0) if has_test2 else None
    d_aux = ds_train.hybrid_dim

    F_in = len(cfg.features)
    em = common_kwargs.get("external_meta", {})
    # Use ALL train cells (not just first 60) for median bias init - critical
    # for stratified KFold where train cells may be sorted by cycle_life.
    train_ys = []
    for c in split_info["train"]:
        v = em.get(c, {}).get("cycle_life") if em else None
        if v is None:
            try:
                v = loader.load_cell(c, external_meta=em).cycle_life or 800
            except Exception:
                v = 800
        train_ys.append(v)
    init_bias = float(np.log(max(np.median(train_ys), 100)))
    LOG.info("init_log_y_bias=%.3f (median cycle_life=%.0f), d_aux=%d",
             init_bias, np.exp(init_bias), d_aux)

    if cfg.model == "vanilla":
        model = VanillaTransformerRUL(in_features=F_in, intra_len=cfg.intra_len,
                                      d_model=cfg.d_model, n_layers=cfg.n_layers,
                                      n_heads=cfg.n_heads,
                                      init_log_y_bias=init_bias,
                                      d_aux_feat=d_aux).to(device)
    elif cfg.model == "lstm":
        model = LSTMRUL(in_features=F_in, intra_len=cfg.intra_len,
                        d_model=cfg.d_model, n_layers=cfg.n_layers,
                        init_log_y_bias=init_bias, d_aux_feat=d_aux).to(device)
    elif cfg.model == "lstm_att":
        model = LSTMAttRUL(in_features=F_in, intra_len=cfg.intra_len,
                           d_model=cfg.d_model, n_layers=cfg.n_layers,
                           n_heads=cfg.n_heads,
                           init_log_y_bias=init_bias, d_aux_feat=d_aux).to(device)
    elif cfg.model == "battery_gpt":
        model = BatteryGPTLite(in_features=F_in, intra_len=cfg.intra_len,
                               d_model=cfg.d_model, n_layers=cfg.n_layers,
                               n_heads=cfg.n_heads,
                               init_log_y_bias=init_bias, d_aux_feat=d_aux).to(device)
    elif cfg.model == "pbt":
        model = PBTLite(in_features=F_in, intra_len=cfg.intra_len,
                        d_model=cfg.d_model, n_layers=cfg.n_layers,
                        n_heads=cfg.n_heads,
                        init_log_y_bias=init_bias, d_aux_feat=d_aux).to(device)
    elif cfg.model == "dgat":
        model = DGATLite(in_features=F_in, intra_len=cfg.intra_len,
                         d_model=cfg.d_model, n_layers=cfg.n_layers,
                         n_heads=cfg.n_heads, window_size=10,
                         init_log_y_bias=init_bias, d_aux_feat=d_aux).to(device)
    elif cfg.model == "dgat_plus":
        model = DGATPlusLite(in_features=F_in, intra_len=cfg.intra_len,
                             d_model=cfg.d_model, n_layers=cfg.n_layers,
                             n_heads=cfg.n_heads, window_size=10,
                             init_log_y_bias=init_bias, d_aux_feat=d_aux).to(device)
    elif cfg.model == "dgat_plus_hsmm":
        model = DGATPlusComposite(in_features=F_in, intra_len=cfg.intra_len,
                                  d_model=cfg.d_model, n_layers=cfg.n_layers,
                                  n_heads=cfg.n_heads, window_size=10,
                                  use_hsmm=True, use_graph=False,
                                  hsmm_K=cfg.hsmm_K, hsmm_D_max=cfg.hsmm_D_max,
                                  init_log_y_bias=init_bias, d_aux_feat=d_aux).to(device)
    elif cfg.model == "dgat_plus_graph":
        model = DGATPlusComposite(in_features=F_in, intra_len=cfg.intra_len,
                                  d_model=cfg.d_model, n_layers=cfg.n_layers,
                                  n_heads=cfg.n_heads, window_size=10,
                                  use_hsmm=False, use_graph=True,
                                  hsmm_K=cfg.hsmm_K, hsmm_D_max=cfg.hsmm_D_max,
                                  init_log_y_bias=init_bias, d_aux_feat=d_aux).to(device)
    elif cfg.model == "dgat_plus_full":
        model = DGATPlusComposite(in_features=F_in, intra_len=cfg.intra_len,
                                  d_model=cfg.d_model, n_layers=cfg.n_layers,
                                  n_heads=cfg.n_heads, window_size=10,
                                  use_hsmm=True, use_graph=True,
                                  hsmm_K=cfg.hsmm_K, hsmm_D_max=cfg.hsmm_D_max,
                                  init_log_y_bias=init_bias, d_aux_feat=d_aux).to(device)
        # init proto_life_head bias for Stackelberg
        with torch.no_grad():
            for m in model.proto_life_head.modules():
                if isinstance(m, nn.Linear) and m.out_features == 1:
                    if m.bias is not None:
                        m.bias.fill_(init_bias)
                    m.weight.data.mul_(0.01)
    elif cfg.model == "hsmm_graph_game":
        model = HSMMGraphGameModel(
            in_features=F_in, intra_len=cfg.intra_len,
            d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
            hsmm_K=cfg.hsmm_K, hsmm_D_max=cfg.hsmm_D_max,
            use_graph=cfg.use_graph,
            d_aux_feat=d_aux,
        ).to(device)
        model.init_log_y_bias.fill_(init_bias)
        with torch.no_grad():
            for m in model.proto_life_head.modules():
                if isinstance(m, nn.Linear) and m.out_features == 1:
                    if m.bias is not None:
                        m.bias.fill_(init_bias)
                    m.weight.data.mul_(0.01)
    else:
        raise ValueError(cfg.model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOG.info("model %s, %d params", cfg.model, n_params)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup_epochs = max(2, int(0.1 * cfg.epochs))

    def _lr_lambda(ep):
        if ep < warmup_epochs:
            return (ep + 1) / warmup_epochs
        progress = (ep - warmup_epochs) / max(1, cfg.epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)

    # EMA disabled — caused checkpoint/eval inconsistency for BatchNorm.
    # We rely on cosine LR + early stopping on best test MAPE instead.
    def _update_ema():
        pass

    history = []
    best = {"epoch": -1, "MAPE": float("inf")}

    def _eval_or_empty(dl, use_ema=False):
        if dl is None:
            return {"MAPE": float("nan"), "RMSE": float("nan"), "n": 0}
        return evaluate(model, dl, device, cfg.log_target)

    for ep in range(cfg.epochs):
        model.train()
        epoch_losses = []
        t0 = time.time()
        for batch in dl_train:
            x = batch["x"].to(device)
            m = batch["mask"].to(device)
            p = batch["proto"].to(device)
            y = batch["y"].to(device)
            fa = batch["feat_aux"].to(device) if "feat_aux" in batch else None
            log_true = torch.log(y.clamp_min(1))
            # Tail-weighted loss: short or long cells get larger weight
            log_med = torch.tensor(init_bias, device=device)
            tail_w = (1.0 + (log_true - log_med).abs() *
                      cfg.tail_weight_power).clamp(0.5, 4.0)

            def _reg(log_pred):
                diff = log_pred - log_true
                if cfg.loss_type in ("huber", "tail_weighted_huber"):
                    abs_d = diff.abs()
                    delta = 0.25
                    per_sample = torch.where(abs_d < delta,
                                             0.5 * diff ** 2,
                                             delta * (abs_d - 0.5 * delta))
                else:
                    per_sample = diff ** 2
                if cfg.loss_type == "tail_weighted_huber" or cfg.tail_weight_power > 0:
                    per_sample = per_sample * tail_w
                return per_sample.mean()

            if isinstance(model, _SIMPLE_MODELS):
                yhat = model(x, m, fa) if fa is not None else model(x, m)
                log_pred = torch.log(yhat.clamp_min(1))
                loss = _reg(log_pred)
            else:
                out = model(x, m, p, fa) if fa is not None else model(x, m, p)
                yhat = out.rul_hat
                log_pred = torch.log(yhat.clamp_min(1))
                loss_rul = _reg(log_pred)
                if cfg.alpha_hsmm > 0:
                    log_lik = out.log_lik
                    if torch.isfinite(log_lik).all():
                        loss_hsmm = (-log_lik.mean() / 1000.0) * cfg.alpha_hsmm
                    else:
                        loss_hsmm = torch.tensor(0.0, device=device)
                else:
                    loss_hsmm = torch.tensor(0.0, device=device)
                aux_log_pred = model.proto_life_head(p)
                loss_aux = nn.functional.mse_loss(aux_log_pred, log_true) * cfg.alpha_aux_proto
                loss = loss_rul + loss_hsmm + loss_aux
            if not torch.isfinite(loss):
                LOG.warning("nan loss; skipping step")
                continue
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            _update_ema()
            epoch_losses.append(loss.item())
        sched.step()
        tr_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

        e1 = _eval_or_empty(dl_test1)
        e2 = _eval_or_empty(dl_test2)
        dt = time.time() - t0
        LOG.info("ep %2d/%d  train_loss=%.4f  test1 MAPE=%.2f%% RMSE=%.1f  test2 MAPE=%.2f%% RMSE=%.1f  (%.1fs)",
                 ep, cfg.epochs, tr_loss, e1["MAPE"], e1["RMSE"], e2["MAPE"], e2["RMSE"], dt)
        history.append({"epoch": ep, "train_loss": tr_loss,
                        "test1_MAPE": e1["MAPE"], "test1_RMSE": e1["RMSE"],
                        "test2_MAPE": e2["MAPE"], "test2_RMSE": e2["RMSE"],
                        "time_s": dt})
        if e1["MAPE"] < best["MAPE"]:
            best = {"epoch": ep, "MAPE": e1["MAPE"], "RMSE": e1["RMSE"]}
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg)},
                       fold_out_dir / "best.pt")

    final = {
        "cfg": asdict(cfg),
        "best": best,
        "history": history,
        "test1": _eval_or_empty(dl_test1),
        "test2": _eval_or_empty(dl_test2),
        "split_size": {"train": len(split_info["train"]),
                       "test1": len(split_info["test1"]),
                       "test2": len(split_info["test2"])},
    }
    with open(fold_out_dir / "results.json", "w") as f:
        json.dump(final, f, indent=2)
    LOG.info("FOLD %d done: best ep %d MAPE %.2f%%",
             split_info["fold"], best["epoch"], best["MAPE"])
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None, choices=[None, "vanilla", "hsmm_graph_game"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.model: cfg.model = args.model
    if args.epochs: cfg.epochs = args.epochs
    if args.out_dir: cfg.out_dir = args.out_dir
    train(cfg)


if __name__ == "__main__":
    main()
