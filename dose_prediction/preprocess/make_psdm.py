"""Step 2/3 of the OpenKBP preprocessing: per-structure signed distance maps (PSDM).

For every structure S of every patient writes Mask_S.nii.gz and
    PSDM_S = (EDT(S) - EDT(not S)) / 100      (Euclidean distances in mm, positive inside S)
Structures a patient does not have are written as all-zero volumes, so every patient has the
same 11 Mask_* + 11 PSDM_* channels. ct.nii.gz and dose.nii.gz are copied alongside.

Reads   <data_dir>/{train,validation,test}-pats/pt_*/
Writes  <data_dir>/{train,validation,test}-pats_preprocess/pt_*/

(Adapted from DoseDiff's PSDM_OpenKBP.py, https://github.com/whisney/DoseDiff, MIT.)
"""
import argparse
import os
import shutil
from functools import partial
from multiprocessing import Pool

import numpy as np
from scipy import ndimage

from nii_utils import SPLITS, nii_read, nii_write

HERE = os.path.dirname(os.path.abspath(__file__))
PSDM_ROIS = ["PTV70", "PTV63", "PTV56", "possible_dose_mask", "Brainstem", "SpinalCord", "RightParotid",
             "LeftParotid", "Esophagus", "Larynx", "Mandible"]


def process_patient(patient, src_split, dst_split):
    src, dst = os.path.join(src_split, patient), os.path.join(dst_split, patient)
    os.makedirs(dst, exist_ok=True)
    _, spacing, origin, direction = nii_read(os.path.join(src, "ct.nii.gz"))
    for name in PSDM_ROIS:
        path = os.path.join(src, f"{name}.nii.gz")
        if os.path.isfile(path):
            shutil.copy(path, os.path.join(dst, f"Mask_{name}.nii.gz"))
            mask, spacing, origin, direction = nii_read(path, as_type=np.uint8)
            dist_in = ndimage.distance_transform_edt(mask, sampling=spacing)
            dist_out = ndimage.distance_transform_edt(1 - mask, sampling=spacing)
            nii_write(os.path.join(dst, f"PSDM_{name}.nii.gz"), (dist_in - dist_out) / 100, spacing, origin, direction)
        else:
            zeros = np.zeros((128, 128, 128), np.uint8)
            nii_write(os.path.join(dst, f"PSDM_{name}.nii.gz"), zeros, spacing, origin, direction)
            nii_write(os.path.join(dst, f"Mask_{name}.nii.gz"), zeros, spacing, origin, direction)
    for name in ("ct", "dose"):
        shutil.copy(os.path.join(src, f"{name}.nii.gz"), os.path.join(dst, f"{name}.nii.gz"))
    return patient


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default=os.path.join(HERE, "..", "data", "preprocessed"))
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    for split in SPLITS:
        src_split = os.path.join(args.data_dir, f"{split}-pats")
        dst_split = os.path.join(args.data_dir, f"{split}-pats_preprocess")
        patients = sorted(os.listdir(src_split))
        with Pool(args.num_workers) as pool:
            for i, pt in enumerate(pool.imap(partial(process_patient, src_split=src_split, dst_split=dst_split),
                                             patients), 1):
                print(f"[{split}] {i}/{len(patients)} {pt}", flush=True)


if __name__ == "__main__":
    main()
