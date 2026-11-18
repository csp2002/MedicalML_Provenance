"""nnU-Net (v2) raw dataset of one tumor dataset: every case cropped along the axial axis to its first..last tumor
slice, label binarised (tumor = 1). Writes <nnunet_raw>/Dataset<ID>_<Name>TO/ with imagesTr/labelsTr (train + val),
imagesTs/labelsTs (test; labelsTs is used only by evaluate.py), dataset.json, crop_ranges.json and
splits_final.json (fold 0 = this repository's train / val split).

    python nnunet/prepare_nnunet.py --dataset colon
"""
import argparse
import json
import os
import sys
import time

import nibabel as nib
import numpy as np

TUMOR_SEG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(TUMOR_SEG_ROOT, "data"))
from prepare_2d import DATASET_CONFIG  # noqa: E402  (axial axis, tumor label, raw folder name)

NNUNET_ID = {"kits": 511, "lits": 512, "pancreas": 513, "colon": 514}
NNUNET_NAME = {"kits": "Dataset511_KiTSTO", "lits": "Dataset512_LiTSTO", "pancreas": "Dataset513_PancreasTO",
               "colon": "Dataset514_ColonTO"}
DEFAULT_NNUNET_RAW = os.path.join(TUMOR_SEG_ROOT, "data", "nnUNet_raw")


def tumor_range(label, axis, target):
    other = tuple(a for a in range(label.ndim) if a != axis)
    z = np.where((label == target).any(axis=other))[0]
    return (int(z[0]), int(z[-1])) if len(z) else None


def convert_case(img_path, seg_path, axis, target, img_out, seg_out):
    seg_img = nib.load(seg_path)
    rng = tumor_range(np.asanyarray(seg_img.dataobj), axis, target)
    if rng is None:
        raise ValueError(f"no tumor in {seg_path}")
    sl = [slice(None)] * 3
    sl[axis] = slice(rng[0], rng[1] + 1)
    img_crop = nib.load(img_path).slicer[tuple(sl)]
    nib.save(img_crop, img_out)
    seg_crop = seg_img.slicer[tuple(sl)]
    binary = (np.rint(seg_crop.get_fdata()).astype(np.int16) == target).astype(np.uint8)
    # The label is written with the geometry of its image (the two are voxel-aligned): a few official LiTS
    # segmentation files (volumes 48-52) carry a wrong 1 x 1 x 1 mm spacing, which nnU-Net rejects.
    header = img_crop.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1, 0)
    nib.save(nib.Nifti1Image(binary, img_crop.affine, header), seg_out)
    return rng


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(NNUNET_ID))
    parser.add_argument("--raw_dir", default=None, help="extracted raw dataset (default: data/raw/<name>)")
    parser.add_argument("--nnunet_raw", default=os.environ.get("nnUNet_raw", DEFAULT_NNUNET_RAW))
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    raw_dir = args.raw_dir or os.path.join(TUMOR_SEG_ROOT, "data", "raw", cfg["raw_dir_name"])
    axis, target = cfg["spatial_index"][0], cfg["target_class"]
    with open(os.path.join(TUMOR_SEG_ROOT, "data", "splits", f"{args.dataset}.json")) as f:
        splits = json.load(f)
    out = os.path.join(args.nnunet_raw, NNUNET_NAME[args.dataset])
    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    t0, n_train, crop_ranges = time.time(), 0, {}
    for split in ("train", "val", "test"):
        img_dir, seg_dir = ("imagesTs", "labelsTs") if split == "test" else ("imagesTr", "labelsTr")
        for case, (img_rel, seg_rel) in splits[split].items():
            rng = convert_case(os.path.join(raw_dir, img_rel), os.path.join(raw_dir, seg_rel), axis, target,
                               os.path.join(out, img_dir, f"{case}_0000.nii.gz"),
                               os.path.join(out, seg_dir, f"{case}.nii.gz"))
            n_train += split != "test"
            crop_ranges[case] = list(rng)
            print(f"[{args.dataset}/{split}] {case}: slices {rng[0]}-{rng[1]}", flush=True)

    with open(os.path.join(out, "dataset.json"), "w") as f:
        json.dump({"channel_names": {"0": "CT"}, "labels": {"background": 0, "tumor": 1}, "numTraining": n_train,
                   "file_ending": ".nii.gz", "name": NNUNET_NAME[args.dataset],
                   "description": f"tumor-only 3D sub-volumes ({args.dataset}); label binarised from {target}"},
                  f, indent=2)
    with open(os.path.join(out, "crop_ranges.json"), "w") as f:  # axial slice range kept per case (inclusive)
        json.dump(crop_ranges, f, indent=1)
    with open(os.path.join(out, "splits_final.json"), "w") as f:
        json.dump([{"train": list(splits["train"]), "val": list(splits["val"])}], f, indent=1)
    print(f"done in {time.time() - t0:.0f}s -> {out} (nnU-Net dataset id {NNUNET_ID[args.dataset]})")


if __name__ == "__main__":
    main()
