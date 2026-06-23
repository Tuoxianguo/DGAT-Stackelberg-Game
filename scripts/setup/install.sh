#!/usr/bin/env bash
# One-shot install of Python deps on the Bohrium server.
# Idempotent: re-runnable; uses pip; pins minimum versions.
# Note: xgboost/lightgbm 等需要 cmake 编译, 用 pre-built wheel 跳过.

set -uo pipefail

echo "[install] python: $(python3 --version)"

# Tsinghua mirror for speed in CN
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
PIP="pip3 install --upgrade --prefer-binary"

echo "[install] ===== core scientific stack ====="
$PIP \
    "numpy>=1.24,<2.0" "pandas>=2.0" "scipy>=1.10" h5py tqdm pyyaml rich \
    "matplotlib>=3.7" "seaborn>=0.12" "scikit-learn>=1.3" "statsmodels>=0.14" \
    "networkx>=3.1" "hmmlearn>=0.3" \
    "pyarrow>=14" fastparquet || echo "[install] WARN: core stack partial fail"

echo "[install] ===== HuggingFace tools for dataset download ====="
$PIP "huggingface_hub>=0.24" "datasets>=2.20" hf_transfer || echo "[install] WARN: HF tools failed"

echo "[install] ===== xgboost / lightgbm (binary only, skip source) ====="
$PIP --only-binary=:all: xgboost lightgbm || echo "[install] WARN: gboost skipped (cmake missing)"
$PIP "optuna>=3.4" || true

echo "[install] ===== torch 2.4 + cu12.1 (compatible with T4) ====="
# pytorch.org direct, official wheels
pip3 install --upgrade --prefer-binary \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.0 torchvision==0.19.0 || \
pip3 install --upgrade --prefer-binary torch==2.4.0 torchvision==0.19.0

echo "[install] ===== PyG (heterogeneous graph) ====="
$PIP "torch_geometric==2.5.3"
# Optional: scatter/sparse compiled wheels (pure-python fallback exists in PyG 2.5)
pip3 install --no-build-isolation --prefer-binary \
    torch_scatter torch_sparse \
    -f https://data.pyg.org/whl/torch-2.4.0+cu121.html 2>/dev/null || \
    echo "[install] NOTE: torch_scatter/sparse wheels not found, will use pure-python ops"

echo "[install] ===== RL and BO ====="
$PIP "stable-baselines3>=2.3" "gymnasium>=0.29" "botorch>=0.11" "gpytorch>=1.12" || true

echo "[install] ===== transformers (only for BatteryGPT-style baselines) ====="
$PIP "transformers>=4.42" "accelerate>=0.30" "safetensors>=0.4" || true

echo "[install] ===== dev / notebook ====="
$PIP jupyterlab ipywidgets pytest pytest-xdist black isort ruff || true

echo "[install] ===== verification ====="
python3 - <<'PY'
import importlib
mods = ["torch", "torch_geometric", "sklearn", "pandas", "numpy",
        "hmmlearn", "networkx", "matplotlib", "stable_baselines3",
        "botorch", "transformers", "huggingface_hub", "pyarrow"]
ok = bad = 0
for m in mods:
    try:
        x = importlib.import_module(m)
        print(f"  OK  {m}=={getattr(x, '__version__', '?')}")
        ok += 1
    except Exception as e:
        print(f"  !!  {m}: {e}")
        bad += 1
print(f"-- {ok} ok / {bad} bad --")
import torch
print("torch cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-gpu")
PY

echo "[install] done"
