"""Inspect the .mat file structure to find the right key names."""
import sys
import h5py
import numpy as np

p = "data/raw/mit_raw/2017-05-12_batchdata_updated_struct_errorcorrect.mat"
with h5py.File(p, "r") as f:
    print("Top keys:", list(f.keys()))
    batch = f["batch"]
    print("batch keys:", list(batch.keys()))
    print()
    # Look at one cell summary
    summ_ref = batch["summary"][0, 0]
    summ = f[summ_ref]
    print("summary keys for cell 0:", list(summ.keys()))
    for k in summ.keys():
        try:
            arr = np.asarray(f[summ[k][0, 0]]).squeeze()
            print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}, first 3 vals={arr.flat[:3]}")
        except Exception as e:
            print(f"  {k}: {e}")
    print()
    # Look at cycles for cell 0
    cy_ref = batch["cycles"][0, 0]
    cy = f[cy_ref]
    print("cycle keys for cell 0:", list(cy.keys()))
    for k in cy.keys():
        try:
            ds = cy[k]
            print(f"  {k}: dtype={ds.dtype}, shape={ds.shape}")
            # Try first cycle
            arr0 = np.asarray(f[ds[0, 0]]).squeeze()
            print(f"     cycle0 shape={arr0.shape}, dtype={arr0.dtype}, first 3 vals={arr0.flat[:3]}")
        except Exception as e:
            print(f"  {k}: ERR {e}")
