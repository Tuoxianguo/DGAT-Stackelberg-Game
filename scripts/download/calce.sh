#!/usr/bin/env bash
# Download CALCE CS2 + CX2 lithium-ion battery data.
# Source: https://web.calce.umd.edu/batteries/data.htm  (separate XLS / ZIP files per cell)
#
# We download the most-used CS2 family (4 cells: 35, 36, 37, 38, each ~50 MB)
# and CX2 family (2 cells: 35, 38) by direct URL.

set -uo pipefail

DEST="${DEST:-data/raw/calce}"
mkdir -p "${DEST}"
cd "${DEST}"

# CALCE direct download URLs (these have been stable for years)
declare -A URLS
URLS["CS2_35.zip"]="https://web.calce.umd.edu/batteries/data/CS2_35.zip"
URLS["CS2_36.zip"]="https://web.calce.umd.edu/batteries/data/CS2_36.zip"
URLS["CS2_37.zip"]="https://web.calce.umd.edu/batteries/data/CS2_37.zip"
URLS["CS2_38.zip"]="https://web.calce.umd.edu/batteries/data/CS2_38.zip"
URLS["CX2_35.zip"]="https://web.calce.umd.edu/batteries/data/CX2_35.zip"
URLS["CX2_38.zip"]="https://web.calce.umd.edu/batteries/data/CX2_38.zip"

for k in "${!URLS[@]}"; do
    u="${URLS[$k]}"
    if [[ -f "$k" ]]; then
        sz=$(stat -c%s "$k")
        if (( sz > 1000000 )); then
            echo "[calce] skip $k ($sz bytes)"
            continue
        fi
    fi
    echo "[calce] downloading $k"
    curl -sL --retry 3 --retry-delay 5 -o "$k" "$u" || echo "[calce] FAIL $k"
done

echo "[calce] unzipping"
for z in *.zip; do
    [[ -f "$z" ]] && unzip -o "$z" > /dev/null 2>&1 && echo "  unzipped $z"
done

echo "[calce] done"
du -sh "${DEST}"
ls "${DEST}" | head
