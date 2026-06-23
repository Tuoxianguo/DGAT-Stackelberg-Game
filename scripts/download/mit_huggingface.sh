#!/usr/bin/env bash
# Download MIT/Stanford-Toyota dataset (Severson 2019, 124 cells)
# from the harmonized HuggingFace mirror bsebench-org/severson-2019
# Each cell -> one parquet file under data/raw/mit_hf/
#
# Re-runnable: HF hub caches already-downloaded files.
# Uses hf-mirror.com for fast access from China.

set -uo pipefail

DEST="${DEST:-data/raw/mit_hf}"
mkdir -p "${DEST}"

# Speed up via hf-mirror, enable transfer accelerator
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=0   # hf_transfer not always available

echo "[mit_hf] ensuring huggingface_hub is installed"
python3 -c "import huggingface_hub" 2>/dev/null || \
    pip3 install --upgrade --prefer-binary \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        "huggingface_hub>=0.24"

echo "[mit_hf] downloading bsebench-org/severson-2019 -> ${DEST}"
python3 - <<PY
import os
from huggingface_hub import snapshot_download

dest = "${DEST}"
print(f"[mit_hf] endpoint = {os.environ.get('HF_ENDPOINT')}")
path = snapshot_download(
    repo_id="bsebench-org/severson-2019",
    repo_type="dataset",
    local_dir=dest,
    local_dir_use_symlinks=False,
    allow_patterns=["*.parquet", "README.md", ".gitattributes"],
    max_workers=8,
)
print(f"[mit_hf] downloaded to {path}")
PY

echo "[mit_hf] generating MANIFEST.json"
python3 - <<PY
import hashlib, json, os
dest = "${DEST}"
files = sorted(f for f in os.listdir(dest) if f.endswith(".parquet"))
m = {"source": "https://huggingface.co/datasets/bsebench-org/severson-2019",
     "schema":  "BSEBench TimeSeriesSchema (BPX-1.1)",
     "n_cells": len(files),
     "files": []}
for f in files:
    p = os.path.join(dest, f)
    h = hashlib.sha256()
    with open(p, "rb") as fp:
        for chunk in iter(lambda: fp.read(1<<20), b""):
            h.update(chunk)
    m["files"].append({"name": f, "size": os.path.getsize(p),
                       "sha256": h.hexdigest()[:16]})
with open(os.path.join(dest, "MANIFEST.json"), "w") as fp:
    json.dump(m, fp, indent=2)
total_mb = sum(x["size"] for x in m["files"]) / (1<<20)
print(f"[mit_hf] {len(files)} cells, total {total_mb:.0f} MB")
PY

echo "[mit_hf] done -> ${DEST}"
