"""Inspect the BSEBench Severson parquet schema after download."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from battery_paper.data import BSEBenchLoader


def main():
    root = "data/raw/mit_hf"
    ldr = BSEBenchLoader(root)
    cells = ldr.list_cells()
    print(f"N cells: {len(cells)}")
    print(f"first 10 cell ids: {cells[:10]}")
    print(f"last 10  cell ids: {cells[-10:]}")
    if not cells:
        print("NO PARQUET FILES FOUND")
        return 1
    cell = ldr.load_cell(cells[0])
    print(f"\n=== cell {cell.cell_id} ===")
    print(f"columns ({len(cell.df.columns)}):", list(cell.df.columns))
    print(f"shape: {cell.df.shape}")
    print(f"dtypes:\n{cell.df.dtypes}")
    print("\nhead:")
    print(cell.df.head(8))
    print(f"\ncycle_life: {cell.cycle_life}")
    print(f"policy: {cell.policy}")
    print(f"chemistry: {cell.chemistry}")
    print(f"protocol parsed: {cell.protocol}")
    # check cycle_index range
    if "cycle_index" in cell.df.columns:
        ci = cell.df["cycle_index"]
        print(f"cycle_index range: {ci.min()}..{ci.max()}, n_unique={ci.nunique()}")
    elif "cycle" in cell.df.columns:
        ci = cell.df["cycle"]
        print(f"cycle range: {ci.min()}..{ci.max()}, n_unique={ci.nunique()}")
    # try summarize
    try:
        sm = ldr.summarize_cell(cells[0])
        print(f"\nsummarize_cell({cells[0]}):")
        print(sm.head(5))
        print(f"  rows: {len(sm)}")
    except Exception as e:
        print(f"summarize_cell failed: {e}")

    # build full manifest preview (without writing)
    print("\n=== quick stats across all cells (loading metadata only) ===")
    print("Sampling first 10 cells policy/cycle_life ...")
    rows = []
    for cid in cells[:10]:
        c = ldr.load_cell(cid)
        rows.append((c.cell_id, c.batch, c.cycle_life, c.policy, c.protocol.get("CC1"),
                     c.protocol.get("CC2")))
    for r in rows:
        print(" ", r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
