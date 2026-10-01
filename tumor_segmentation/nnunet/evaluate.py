"""Score nnU-Net test predictions with the same slice-level metrics as the 2D baselines.

Every axial slice of a test case that contains tumor in the ground truth is scored (Dice / IoU at the original
resolution, see common/metrics.py), i.e. exactly the slices of data/2D/<dataset>/test. The 3D Dice over the
whole tumor-only sub-volume is reported as well (`dice3d_mean`).

    python nnunet/evaluate.py --dataset colon --pred_dir runs/nnunet_colon/pred_test --out_dir runs/nnunet_colon/eval_test
"""
import argparse
import json
import os
import sys

import nibabel as nib
import numpy as np

TUMOR_SEG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TUMOR_SEG_ROOT)
sys.path.insert(0, os.path.join(TUMOR_SEG_ROOT, "data"))

from common.metrics import EPS, slice_scores, summarize, write_results  # noqa: E402
from prepare_2d import DATASET_CONFIG  # noqa: E402

from prepare_nnunet import DEFAULT_NNUNET_RAW, NNUNET_NAME  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(NNUNET_NAME))
    parser.add_argument("--pred_dir", required=True, help="output folder of nnUNetv2_predict")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--nnunet_raw", default=os.environ.get("nnUNet_raw", DEFAULT_NNUNET_RAW))
    parser.add_argument("--checkpoint", default="", help="recorded in metrics.json (e.g. the nnU-Net results folder)")
    args = parser.parse_args()

    axis = DATASET_CONFIG[args.dataset]["spatial_index"][0]
    label_dir = os.path.join(args.nnunet_raw, NNUNET_NAME[args.dataset], "labelsTs")
    with open(os.path.join(TUMOR_SEG_ROOT, "data", "splits", f"{args.dataset}.json")) as f:
        cases = list(json.load(f)["test"])
    ranges_path = os.path.join(args.nnunet_raw, NNUNET_NAME[args.dataset], "crop_ranges.json")
    if not os.path.isfile(ranges_path):
        raise FileNotFoundError(f"{ranges_path} not found -- is --nnunet_raw the folder written by prepare_nnunet.py?")
    with open(ranges_path) as f:
        offsets = {c: r[0] for c, r in json.load(f).items()}  # first kept axial slice of every case
    rows, dice3d = [], []
    for case in cases:
        pred_path = os.path.join(args.pred_dir, f"{case}.nii.gz")
        if not os.path.isfile(pred_path):
            raise FileNotFoundError(f"missing prediction {pred_path}")
        gt = np.moveaxis(np.rint(nib.load(os.path.join(label_dir, f"{case}.nii.gz")).get_fdata()) == 1, axis, 0)
        pred = np.moveaxis(np.rint(nib.load(pred_path).get_fdata()) == 1, axis, 0)
        assert pred.shape == gt.shape, f"{case}: prediction {pred.shape} vs label {gt.shape}"
        inter = np.logical_and(pred, gt).sum()
        dice3d.append(float((2 * inter + EPS) / (pred.sum() + gt.sum() + EPS)))
        for z in range(gt.shape[0]):
            if gt[z].any():  # tumor slices only, as for the 2D baselines
                # named like the 2D slices (original axial index), so per_slice.csv files can be joined
                rows.append(dict(case=case, slice=f"{z + offsets[case]:03d}.png", **slice_scores(pred[z], gt[z])))
        print(f"{case}: 3D Dice {dice3d[-1]:.4f}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    summary = dict(model="nnunet_3d_fullres", dataset=args.dataset, split="test", checkpoint=args.checkpoint,
                   **summarize(rows), dice3d_mean=float(np.mean(dice3d)))
    write_results(rows, summary, os.path.join(args.out_dir, "per_slice.csv"), os.path.join(args.out_dir, "metrics.json"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
