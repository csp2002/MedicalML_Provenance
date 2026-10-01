"""2D slice dataset of the preprocessed OpenKBP data (output of preprocess/nii_to_npy.py).

Each sample is one axial slice: (ct [1,128,128], condition [20,128,128], dose [1,128,128]),
CT and dose already scaled to [-1, 1]. Training augmentation (DoseDiff): with p=0.8 apply
shift/scale/rotate (p=0.3; shift 0.1, scale 0.1, rotate +-90 deg, nearest, replicate border),
horizontal flip (p=0.3) and vertical flip (p=0.3) jointly to all channels.
"""
import os
import warnings

import cv2
import numpy as np
import torch
from albumentations import Compose, HorizontalFlip, ShiftScaleRotate, VerticalFlip
from torch.utils.data import Dataset

from model import MASK_CHANNELS, PSDM_CHANNELS, PTV_WEIGHTS

FIELDS = ["ct", "dose"] + list(PTV_WEIGHTS) + MASK_CHANNELS + PSDM_CHANNELS
# DoseDiff's original augmentation; albumentations>=2 suggests Affine instead but ShiftScaleRotate still works.
warnings.filterwarnings("ignore", message="ShiftScaleRotate is a special case of Affine")


class OpenKBPSliceDataset(Dataset):
    def __init__(self, data_root, augment=False):
        self.data_root = data_root
        self.files = sorted(os.listdir(os.path.join(data_root, "ct")))
        self.transforms = Compose([
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=90, p=0.3,
                             border_mode=cv2.BORDER_REPLICATE, interpolation=cv2.INTER_NEAREST),
            HorizontalFlip(p=0.3), VerticalFlip(p=0.3)], p=0.8) if augment else None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        f = {k: np.load(os.path.join(self.data_root, k, name))[:, :, np.newaxis] for k in FIELDS}
        ptv = sum(w * f[k] for k, w in PTV_WEIGHTS.items())
        data = np.concatenate([f["ct"], f["dose"], ptv] + [f[k] for k in MASK_CHANNELS] +
                              [f[k] for k in PSDM_CHANNELS], axis=-1)
        if self.transforms is not None:
            data = self.transforms(image=data)["image"]
        data = torch.from_numpy(np.ascontiguousarray(data)).permute(2, 0, 1)
        return data[0:1], data[2:], data[1:2]  # ct, condition (20 ch), dose
