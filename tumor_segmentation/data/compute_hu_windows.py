"""Compute the HU window of a dataset from its training cases.

The window is the 0.5 / 99.5 percentile of the CT intensities of all tumor voxels of the training cases
(``train`` in ``splits/<dataset>.json``; validation and test cases are not used), rounded to whole HU.
The windows in ``prepare_2d.py`` (``DATASET_CONFIG[...]["intensity_range"]``) were computed with this script.

Example:
    python data/compute_hu_windows.py --dataset colon --raw_dir data/raw/Task10_Colon
"""
import argparse
import json
import os
from functools import partial
from multiprocessing import Pool

import nibabel as nib
import numpy as np

from prepare_2d import DATASET_CONFIG, HERE


def tumor_voxels(paths, raw_dir, target_class):
    img_rel, seg_rel = paths
    img = nib.load(os.path.join(raw_dir, img_rel)).get_fdata(dtype=np.float32)
    seg = nib.load(os.path.join(raw_dir, seg_rel)).get_fdata(dtype=np.float32)
    return img[seg == target_class]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIG))
    parser.add_argument("--raw_dir", default=None,
                        help="Folder of the extracted raw dataset (default: data/raw/<kits23|LITS17|Task07_Pancreas|Task10_Colon>)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of cases loaded in parallel")
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    raw_dir = args.raw_dir or os.path.join(HERE, "raw", cfg["raw_dir_name"])
    with open(os.path.join(HERE, "splits", f"{args.dataset}.json")) as f:
        train_cases = json.load(f)["train"]
    with Pool(args.num_workers) as pool:
        voxels = np.concatenate(pool.map(partial(tumor_voxels, raw_dir=raw_dir, target_class=cfg["target_class"]),
                                         list(train_cases.values())))
    lo, hi = np.percentile(voxels, (0.5, 99.5))
    print(f"{args.dataset}: {len(train_cases)} training cases, {len(voxels)} tumor voxels, "
          f"0.5 / 99.5 percentiles {lo:.1f} / {hi:.1f} HU -> intensity_range=({round(lo)}, {round(hi)}) "
          f"(in prepare_2d.py: {cfg['intensity_range']})")


if __name__ == "__main__":
    main()
