"""Step 3/3 of the OpenKBP preprocessing: 3D volumes -> normalised 2D axial slices (.npy) for training.

Only slices where the possible-dose mask is non-empty are kept. Normalisation:
    CT:   clip to [0, 2500], then x / 1250 - 1   -> [-1, 1]
    dose: clip to [0, 80] Gy, then x / 40 - 1    -> [-1, 1]
    Mask_* and PSDM_* channels are stored unchanged.

Reads   <data_dir>/{train,validation}-pats_preprocess/pt_*/
Writes  <data_dir>/NPY/{train,validation}/<field>/<pt>_<slice>.npy   (24 fields)

The test split is evaluated directly from the 3D volumes (dosediff/test.py), so it is not
converted unless requested with --splits train validation test.

(Adapted from DoseDiff's NII_to_npy.py, https://github.com/whisney/DoseDiff, MIT.)
"""
import argparse
import os
from functools import partial
from multiprocessing import Pool

import numpy as np

from nii_utils import nii_read

HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = ["ct", "dose", "Mask_Brainstem", "Mask_Esophagus", "Mask_Larynx", "Mask_LeftParotid", "Mask_Mandible",
          "Mask_possible_dose_mask", "Mask_PTV56", "Mask_PTV63", "Mask_PTV70", "Mask_RightParotid", "Mask_SpinalCord",
          "PSDM_Brainstem", "PSDM_Esophagus", "PSDM_Larynx", "PSDM_LeftParotid", "PSDM_Mandible",
          "PSDM_possible_dose_mask", "PSDM_PTV56", "PSDM_PTV63", "PSDM_PTV70", "PSDM_RightParotid", "PSDM_SpinalCord"]


def process_patient(patient, src_split, dst_split):
    src = os.path.join(src_split, patient)
    body, _, _, _ = nii_read(os.path.join(src, "Mask_possible_dose_mask.nii.gz"))
    keep = [i for i in range(body.shape[0]) if body[i].max() > 0]
    for name in FIELDS:
        volume, _, _, _ = nii_read(os.path.join(src, f"{name}.nii.gz"))
        if name == "ct":
            volume = np.clip(volume, 0, 2500) / 1250 - 1
        elif name == "dose":
            volume = np.clip(volume, 0, 80) / 40 - 1
        for i in keep:
            np.save(os.path.join(dst_split, name, f"{patient}_{i}.npy"), volume[i])
    return patient, len(keep)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default=os.path.join(HERE, "..", "data", "preprocessed"))
    parser.add_argument("--splits", nargs="+", default=["train", "validation"], choices=["train", "validation", "test"])
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    for split in args.splits:
        src_split = os.path.join(args.data_dir, f"{split}-pats_preprocess")
        dst_split = os.path.join(args.data_dir, "NPY", split)
        for name in FIELDS:
            os.makedirs(os.path.join(dst_split, name), exist_ok=True)
        patients = sorted(os.listdir(src_split))
        n_slices = 0
        with Pool(args.num_workers) as pool:
            for i, (pt, n) in enumerate(pool.imap(partial(process_patient, src_split=src_split, dst_split=dst_split),
                                                  patients), 1):
                n_slices += n
                print(f"[{split}] {i}/{len(patients)} {pt}: {n} slices", flush=True)
        print(f"[{split}] {len(patients)} patients, {n_slices} slices -> {dst_split}")


if __name__ == "__main__":
    main()
