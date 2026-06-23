#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
for m in battery_gpt pbt dgat; do
    echo "=== Ensemble for $m ==="
    PYTHONPATH=src MODEL_TAG=$m MS_ROOT=experiments/v6_$m SEEDS=42,7,2026 \
        python3 scripts/run/ensemble_multi_seed.py \
        --out experiments/v6_${m}_ens.json 2>&1 | tail -n 10
    echo
done
