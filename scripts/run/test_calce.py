"""Quick test of CALCE loader."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from battery_paper.data.calce_loader import load_all_calce

cells = load_all_calce("data/raw/calce", n_cycles=100, intra_len=64)
print(f"loaded {len(cells)} CALCE cells")
for c in cells:
    import numpy as np
    print(f"  {c['cell_id']}: cycle_life={float(c['y']):.0f}  "
          f"valid cycles={int(c['mask'].sum())}  chem={c['chem']}  "
          f"x range=[{c['x'].min():.3f}, {c['x'].max():.3f}]")
