"""Convert raw 3D CT volumes (NIfTI) into the 2D slice dataset used by every
tumor-segmentation baseline in this repository.

For each case listed in ``splits/<dataset>.json`` the script
  1. loads the CT volume and its label map,
  2. reorders the axes so that axis 0 is the axial (slice) axis,
  3. binarises the label map to the tumor class of the dataset,
  4. clips the CT intensities to a dataset-specific window and rescales to [0, 1],
  5. writes every axial slice that contains tumor as a pair of PNG files
     ``<out_dir>/<split>/CT/<case>/<slice>.png`` and ``<out_dir>/<split>/Mask/<case>/<slice>.png``.

It also writes ``<out_dir>/<split>_slices.csv`` (one row per saved slice) and
``<out_dir>/prepare_config.json`` (the exact parameters used), so the lineage of
every training slice can be traced back to its source volume.

Example:
    python data/prepare_2d.py --dataset colon --raw_dir data/raw/Task10_Colon
"""
import argparse
import csv
import hashlib
import json
import os
import time
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# Per-dataset preprocessing parameters.
#   intensity_range: HU window = 0.5 / 99.5 percentiles of the CT values of the tumor voxels of the training
#                    cases, rounded to whole HU (computed by compute_hu_windows.py; val / test cases not used)
#   spatial_index:   axis permutation that brings the axial axis to position 0
#   target_class:    label value of the tumor in the raw label map
DATASET_CONFIG = {
    "kits": dict(intensity_range=(-45, 227), spatial_index=(0, 1, 2), target_class=2, raw_dir_name="kits23"),
    "lits": dict(intensity_range=(-52, 167), spatial_index=(2, 1, 0), target_class=2, raw_dir_name="LITS17"),
    "pancreas": dict(intensity_range=(-37, 205), spatial_index=(2, 1, 0), target_class=2, raw_dir_name="Task07_Pancreas"),
    "colon": dict(intensity_range=(-31, 171), spatial_index=(2, 1, 0), target_class=1, raw_dir_name="Task10_Colon"),
}


def case_name_from_path(dataset, img_path):
    # KiTS23 keeps one folder per case (case_00000/imaging.nii.gz); the other
    # datasets encode the case id in the file name (colon_001.nii.gz, volume-8.nii).
    if dataset == "kits":
        return os.path.basename(os.path.dirname(img_path))
    return os.path.basename(img_path).split(".")[0]


def process_case(job):
    dataset, split, img_path, seg_path, out_dir = job
    cfg = DATASET_CONFIG[dataset]
    lo, hi = cfg["intensity_range"]
    img = nib.load(img_path).get_fdata().astype(np.float32).transpose(cfg["spatial_index"])
    seg = nib.load(seg_path).get_fdata().astype(np.float32).transpose(cfg["spatial_index"])
    assert img.shape == seg.shape, f"shape mismatch for {img_path}: {img.shape} vs {seg.shape}"
    img[np.isnan(img)] = 0
    seg[np.isnan(seg)] = 0
    seg = (seg == cfg["target_class"]).astype(np.float32)
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo)

    case = case_name_from_path(dataset, img_path)
    ct_dir = os.path.join(out_dir, split, "CT", case)
    mask_dir = os.path.join(out_dir, split, "Mask", case)
    os.makedirs(ct_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    rows = []
    for j in range(img.shape[0]):
        seg_slice = seg[j]
        n_tumor = int(seg_slice.sum())
        if n_tumor == 0:  # only slices that contain tumor are kept
            continue
        # plt.imsave rescales each slice to its own [min, max] and writes an RGBA PNG;
        # the loaders read it back as 8-bit grayscale. This is how the original project wrote its
        # slices; only the HU windows differ (computed from the training cases here).
        plt.imsave(os.path.join(ct_dir, f"{j:03d}.png"), img[j], cmap="gray")
        plt.imsave(os.path.join(mask_dir, f"{j:03d}.png"), seg_slice, cmap="gray")
        rows.append(dict(split=split, case=case, slice_index=j,
                         image=os.path.join(split, "CT", case, f"{j:03d}.png"),
                         mask=os.path.join(split, "Mask", case, f"{j:03d}.png"),
                         height=seg_slice.shape[0], width=seg_slice.shape[1], tumor_pixels=n_tumor,
                         source_image=img_path, source_label=seg_path))
    return case, rows


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIG))
    parser.add_argument("--raw_dir", default=None,
                        help="Folder of the extracted raw dataset (default: data/raw/<kits23|LITS17|Task07_Pancreas|Task10_Colon>)")
    parser.add_argument("--out_dir", default=None, help="Output folder (default: data/2D/<dataset>)")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--num_workers", type=int, default=4, help="Number of cases processed in parallel")
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    raw_dir = args.raw_dir or os.path.join(HERE, "raw", cfg["raw_dir_name"])
    out_dir = args.out_dir or os.path.join(HERE, "2D", args.dataset)
    split_file = os.path.join(HERE, "splits", f"{args.dataset}.json")
    with open(split_file) as f:
        splits = json.load(f)
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"raw dataset folder not found: {raw_dir} (see tumor_segmentation/README.md)")
    os.makedirs(out_dir, exist_ok=True)

    for split in args.splits:
        cases = splits[split]
        jobs = []
        for case_id, (img_rel, seg_rel) in cases.items():
            img_path = os.path.join(raw_dir, img_rel)
            seg_path = os.path.join(raw_dir, seg_rel)
            for p in (img_path, seg_path):
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"{p} (case {case_id}) is missing -- is --raw_dir correct?")
            jobs.append((args.dataset, split, img_path, seg_path, out_dir))
        t0 = time.time()
        all_rows = []
        with Pool(args.num_workers) as pool:
            for i, (case, rows) in enumerate(pool.imap(process_case, jobs), 1):
                all_rows.extend(rows)
                print(f"[{args.dataset}/{split}] {i}/{len(jobs)} {case}: {len(rows)} tumor slices", flush=True)
        csv_path = os.path.join(out_dir, f"{split}_slices.csv")
        fields = ["split", "case", "slice_index", "image", "mask", "height", "width", "tumor_pixels",
                  "source_image", "source_label"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"[{args.dataset}/{split}] {len(jobs)} cases, {len(all_rows)} slices, "
              f"{time.time() - t0:.0f}s -> {csv_path}")

    config_path = os.path.join(out_dir, "prepare_config.json")
    done_splits = list(args.splits)
    if os.path.isfile(config_path):  # keep splits converted by earlier runs
        with open(config_path) as f:
            done_splits = sorted(set(json.load(f).get("splits", [])) | set(args.splits))
    config = dict(dataset=args.dataset, raw_dir=os.path.abspath(raw_dir), out_dir=os.path.abspath(out_dir),
                  splits=done_splits, split_file=os.path.abspath(split_file), split_file_sha256=sha256(split_file),
                  intensity_range=list(cfg["intensity_range"]), spatial_index=list(cfg["spatial_index"]),
                  target_class=cfg["target_class"], kept_slices="axial slices with >=1 tumor pixel",
                  created=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
