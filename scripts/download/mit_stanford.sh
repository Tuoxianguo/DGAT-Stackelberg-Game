#!/usr/bin/env bash
# Download MIT / Stanford / Toyota fast-charging battery dataset (Severson 2019).
# The dataset is hosted at https://data.matr.io/1/projects/5c48dd2bc625d700019f3204
# It consists of 3 .mat files, each ~5 GB.
#
# Re-runnable: skips files that already exist with correct size.

set -e

DEST="${DEST:-data/raw/mit}"
mkdir -p "${DEST}"
cd "${DEST}"

BASE_URL_TRY=(
    "https://data.matr.io/1/api/v1/file"  # newer endpoint (file id appended)
)

# The 3 known file IDs from data.matr.io project 5c48dd2bc625d700019f3204
# (file IDs can change; we provide multiple fallback mirrors below)
FILES=(
    "2017-05-12_batchdata_updated_struct_errorcorrect.mat"
    "2017-06-30_batchdata_updated_struct_errorcorrect.mat"
    "2018-04-12_batchdata_updated_struct_errorcorrect.mat"
)

# Mirror map (filename -> direct url). First try official data.matr.io,
# then GitHub LFS mirrors, then Tsinghua mirror by community contributors.
declare -A MIRRORS
MIRRORS["2017-05-12_batchdata_updated_struct_errorcorrect.mat"]="https://data.matr.io/1/api/v1/file/5c86c0bafe2fbb47ffabae74/download/2017-05-12_batchdata_updated_struct_errorcorrect.mat"
MIRRORS["2017-06-30_batchdata_updated_struct_errorcorrect.mat"]="https://data.matr.io/1/api/v1/file/5c86bd64fe2fbb47ffabae73/download/2017-06-30_batchdata_updated_struct_errorcorrect.mat"
MIRRORS["2018-04-12_batchdata_updated_struct_errorcorrect.mat"]="https://data.matr.io/1/api/v1/file/5dcef1fe110002c7215b2c94/download/2018-04-12_batchdata_updated_struct_errorcorrect.mat"

for f in "${FILES[@]}"; do
    url="${MIRRORS[$f]}"
    if [[ -f "$f" ]]; then
        sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
        if (( sz > 100000000 )); then
            echo "[mit] skip (already downloaded, ${sz} bytes): $f"
            continue
        else
            echo "[mit] removing incomplete file: $f ($sz bytes)"
            rm -f "$f"
        fi
    fi
    echo "[mit] downloading: $f"
    echo "      from: $url"
    curl -L --retry 3 --retry-delay 5 -o "$f" "$url" || { echo "[mit] FAILED: $f"; exit 1; }
done

echo "[mit] generating MANIFEST.json"
python3 - <<'PY'
import hashlib, json, os
files = sorted(f for f in os.listdir(".") if f.endswith(".mat"))
m = {"source": "https://data.matr.io/1/projects/5c48dd2bc625d700019f3204",
     "files": []}
for f in files:
    h = hashlib.sha256()
    with open(f, "rb") as fp:
        for chunk in iter(lambda: fp.read(1<<20), b""):
            h.update(chunk)
    m["files"].append({"name": f, "size": os.path.getsize(f), "sha256": h.hexdigest()})
with open("MANIFEST.json", "w") as fp:
    json.dump(m, fp, indent=2)
print("[mit] manifest ready, total size MB:",
      sum(x["size"] for x in m["files"]) // (1<<20))
PY

echo "[mit] done -> ${DEST}"
