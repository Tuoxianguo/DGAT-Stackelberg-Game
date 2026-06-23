#!/usr/bin/env bash
# Look at the BSEBench dataset README + metadata
echo "=== README ==="
ls -la data/raw/mit_hf/
echo
if [[ -f data/raw/mit_hf/README.md ]]; then
    head -n 200 data/raw/mit_hf/README.md
fi
echo
echo "=== one parquet metadata (b1c0 last rows) ==="
python3 - <<'PY'
import pandas as pd
df = pd.read_parquet("data/raw/mit_hf/b1c0.parquet")
print("shape:", df.shape)
print("columns:", list(df.columns))
print("\ndtypes:", df.dtypes.to_dict())
print("\nlast 3 rows:")
print(df.tail(3))
print("\nsoh_truth non-null count:", df["soh_truth"].notna().sum())
print("step_id unique:", df["step_id"].unique()[:20])
print("cycle_number range:", df["cycle_number"].min(), "..", df["cycle_number"].max())
# find rows where soh_truth is defined (cycle-summary rows?)
soh = df.dropna(subset=["soh_truth"])
print("rows with soh_truth:", len(soh))
print(soh.head())
PY
