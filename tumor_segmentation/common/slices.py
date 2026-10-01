"""Listing and loading of the 2D slice dataset written by data/prepare_2d.py.

Layout: <data_root>/<dataset>/<split>/{CT,Mask}/<case>/<slice>.png
"""
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

DATASETS = ("kits", "lits", "pancreas", "colon")
TUMOR_SEG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_ROOT = os.path.join(TUMOR_SEG_ROOT, "data", "2D")


@dataclass(frozen=True)
class SliceItem:
    case: str
    name: str        # slice file name, e.g. "057.png"
    image_path: str
    mask_path: str


def list_slices(data_root, dataset, split):
    """All (CT, mask) slice pairs of one split, in a deterministic (sorted) order."""
    split_dir = os.path.join(data_root, dataset, split)
    ct_dir = os.path.join(split_dir, "CT")
    if not os.path.isdir(ct_dir):
        raise FileNotFoundError(f"{ct_dir} not found -- run data/prepare_2d.py --dataset {dataset} first")
    items = []
    for case in sorted(os.listdir(ct_dir)):
        case_dir = os.path.join(ct_dir, case)
        if not os.path.isdir(case_dir):
            continue
        for name in sorted(os.listdir(case_dir)):
            if not name.endswith(".png"):
                continue
            image_path = os.path.join(case_dir, name)
            mask_path = os.path.join(split_dir, "Mask", case, name)
            if not os.path.isfile(mask_path):
                raise FileNotFoundError(f"missing mask for {image_path}: {mask_path}")
            items.append(SliceItem(case, name, image_path, mask_path))
    if not items:
        raise RuntimeError(f"no slices found under {split_dir}")
    return items


def load_image(path):
    """8-bit grayscale CT slice as float32 in [0, 255]."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def load_mask(path):
    """Binary tumor mask (uint8, values {0, 1})."""
    return (np.asarray(Image.open(path).convert("L")) > 127).astype(np.uint8)
