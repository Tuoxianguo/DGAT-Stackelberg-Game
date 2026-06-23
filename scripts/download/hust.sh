#!/usr/bin/env bash
# Download the HUST battery dataset (Tian et al. Energy 2022).
# Hosted on Zenodo at DOI 10.5281/zenodo.6405084.
#
# Re-runnable: skips files already there.

set -uo pipefail

DEST="${DEST:-data/raw/hust}"
mkdir -p "${DEST}"
cd "${DEST}"

# Zenodo deposit 6405084 (latest version as of writing); fetch all .zip
echo "[hust] fetching file listing from Zenodo API"
META=$(curl -sL https://zenodo.org/api/records/6405084)
echo "$META" | python3 -c "
import json, sys, urllib.request, os
d = json.load(sys.stdin)
files = d.get('files', [])
print(f'[hust] {len(files)} files in deposit')
for f in files:
    url = f['links']['self']
    name = f['key']
    size = f['size']
    print(f'  {name} ({size/1e6:.0f} MB)  -> {url}')
    out = name
    if os.path.exists(out) and os.path.getsize(out) >= size:
        print(f'    [skip] already downloaded')
        continue
    urllib.request.urlretrieve(url, out)
    print(f'    [ok] {out}')
"

echo "[hust] unzipping any .zip files"
for z in *.zip; do
    if [[ -f "$z" ]]; then
        unzip -o "$z" > /dev/null && echo "  unzipped $z"
    fi
done

echo "[hust] done"
du -sh "${DEST}"
ls "${DEST}" | head
