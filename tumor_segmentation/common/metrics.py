"""Evaluation metrics shared by all tumor-segmentation baselines.

Metrics are computed per slice at the original slice resolution:
    Dice = (2|P∩G| + eps) / (|P| + |G| + eps),   IoU = (|P∩G| + eps) / (|P∪G| + eps),  eps = 1e-6
and reported as the mean over all test slices (every test slice contains tumor).
A case-level Dice/IoU (intersection and union pooled over the tumor slices of a case)
is reported as a secondary summary.
"""
import csv
import json
from collections import defaultdict

import numpy as np

EPS = 1e-6


def slice_scores(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = int(np.logical_and(pred, gt).sum())
    p, g = int(pred.sum()), int(gt.sum())
    union = p + g - inter
    return dict(dice=(2 * inter + EPS) / (p + g + EPS), iou=(inter + EPS) / (union + EPS),
                intersection=inter, pred_pixels=p, gt_pixels=g)


def summarize(rows):
    """rows: list of dicts with keys case, dice, iou, intersection, pred_pixels, gt_pixels."""
    per_case = defaultdict(lambda: np.zeros(3, dtype=np.int64))
    for r in rows:
        per_case[r["case"]] += (r["intersection"], r["pred_pixels"], r["gt_pixels"])
    case_dice = [(2 * i + EPS) / (p + g + EPS) for i, p, g in per_case.values()]
    case_iou = [(i + EPS) / (p + g - i + EPS) for i, p, g in per_case.values()]
    return dict(
        num_slices=len(rows),
        num_cases=len(per_case),
        slice_dice_mean=float(np.mean([r["dice"] for r in rows])),
        slice_dice_std=float(np.std([r["dice"] for r in rows])),
        slice_iou_mean=float(np.mean([r["iou"] for r in rows])),
        slice_iou_std=float(np.std([r["iou"] for r in rows])),
        case_dice_mean=float(np.mean(case_dice)),
        case_iou_mean=float(np.mean(case_iou)),
    )


def write_results(rows, summary, csv_path, json_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
