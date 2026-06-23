#!/usr/bin/env bash
# Download the Tier 1 raw mirror of Severson 2019 .mat files (8.3 GB)
# from HuggingFace bsebench-org/severson-2019-raw.
# Needed for the cycle_life + policy_readable metadata fields that
# the canonical Tier 2 parquet does not include.

set -uo pipefail

DEST="${DEST:-data/raw/mit_raw}"
mkdir -p "${DEST}"

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "[mit_raw] ensuring huggingface_hub installed"
python3 -c "import huggingface_hub" 2>/dev/null || \
    pip3 install --upgrade --prefer-binary \
        -i https://pypi.tuna.tsinghua.edu.cn/simple "huggingface_hub>=0.24"

echo "[mit_raw] downloading bsebench-org/severson-2019-raw -> ${DEST}"
python3 - <<PY
import os
from huggingface_hub import snapshot_download
print("endpoint =", os.environ.get("HF_ENDPOINT"))
path = snapshot_download(
    repo_id="bsebench-org/severson-2019-raw",
    repo_type="dataset",
    local_dir="${DEST}",
    allow_patterns=["*.mat", "README.md"],
    max_workers=4,
)
print("downloaded to", path)
import os
for f in sorted(os.listdir("${DEST}")):
    p = os.path.join("${DEST}", f)
    if os.path.isfile(p):
        print(f, os.path.getsize(p))
PY

echo "[mit_raw] done"
