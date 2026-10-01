"""Step 1/3 of the OpenKBP preprocessing: sparse CSV files -> 128^3 NIfTI volumes.

Reads   <provided_data>/{train,validation,test}-pats/pt_*/{ct,dose,<roi>,voxel_dimensions}.csv
Writes  <out_dir>/{train,validation,test}-pats/pt_*/{ct,dose,<roi>}.nii.gz

(Adapted from DoseDiff's csv2nii.py, https://github.com/whisney/DoseDiff, MIT.)
"""
import argparse
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd

from nii_utils import ALL_ROIS, SPLITS, nii_write

HERE = os.path.dirname(os.path.abspath(__file__))
ORIGIN = (0, 0, 0)
DIRECTION = [1, 0, 0, 0, 1, 0, 0, 0, 1]


def load_csv(path):
    """OpenKBP CSV: a sparse volume (voxel index + value) or, for masks, a list of voxel indices."""
    df = pd.read_csv(path, index_col=0)
    if df.isnull().values.any():  # mask: only indices
        return np.array(df.index).squeeze()
    return {"indices": df.index.values, "data": df.data.values}


def to_volume(data, is_mask):
    volume = np.zeros((128, 128, 128, 1))
    if is_mask:
        np.put(volume, data, 1)
    else:
        np.put(volume, data["indices"], data["data"])
    # OpenKBP stores (x, y, z); reorder to (z, x, y) and flip the slice axis
    return np.transpose(volume.squeeze(), (2, 0, 1))[::-1, :, :]


def convert_patient(patient, split_dir, out_split_dir):
    src = os.path.join(split_dir, patient)
    dst = os.path.join(out_split_dir, patient)
    os.makedirs(dst, exist_ok=True)
    spacing = np.loadtxt(os.path.join(src, "voxel_dimensions.csv"))[[2, 0, 1]]
    for roi in ALL_ROIS:  # not every patient has every structure
        if os.path.isfile(os.path.join(src, f"{roi}.csv")):
            volume = to_volume(load_csv(os.path.join(src, f"{roi}.csv")), is_mask=True)
            nii_write(os.path.join(dst, f"{roi}.nii.gz"), volume, spacing, ORIGIN, DIRECTION, as_type=np.uint8)
    for name in ("ct", "dose"):
        volume = to_volume(load_csv(os.path.join(src, f"{name}.csv")), is_mask=False)
        nii_write(os.path.join(dst, f"{name}.nii.gz"), volume, spacing, ORIGIN, DIRECTION)
    return patient


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provided_data", default=os.path.join(HERE, "..", "data", "open-kbp", "provided-data"))
    parser.add_argument("--out_dir", default=os.path.join(HERE, "..", "data", "preprocessed"))
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    for split in SPLITS:
        split_dir = os.path.join(args.provided_data, f"{split}-pats")
        out_split_dir = os.path.join(args.out_dir, f"{split}-pats")
        patients = sorted(p for p in os.listdir(split_dir) if p.startswith("pt_"))
        with Pool(args.num_workers) as pool:
            for i, pt in enumerate(pool.imap(partial(convert_patient, split_dir=split_dir, out_split_dir=out_split_dir),
                                             patients), 1):
                print(f"[{split}] {i}/{len(patients)} {pt}", flush=True)


if __name__ == "__main__":
    main()
