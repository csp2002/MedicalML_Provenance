"""Slice dataset for the 2D CNN / ViT baselines (UNet++, TransUNet).

Each CT slice is min-max normalised to [0, 1] and resized to img_size x img_size
(cubic for the image, nearest for the mask). Training augmentation follows TransUNet:
with p=0.5 a random 90-degree rotation + flip, otherwise with p=0.5 a random rotation
in [-20, 20) degrees (0.25 overall).
"""
import random

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset

from common.slices import list_slices, load_image, load_mask


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    axis = np.random.randint(0, 2)
    image = np.flip(np.rot90(image, k), axis=axis).copy()
    label = np.flip(np.rot90(label, k), axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False, mode="nearest")
    label = ndimage.rotate(label, angle, order=0, reshape=False, mode="nearest")
    return image, label


def preprocess_image(image, img_size):
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    h, w = image.shape
    if (h, w) != (img_size, img_size):
        image = ndimage.zoom(image, (img_size / h, img_size / w), order=3)
    return image


class SliceDataset(Dataset):
    def __init__(self, data_root, dataset, split, img_size=224, augment=False):
        self.items = list_slices(data_root, dataset, split)
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = preprocess_image(load_image(item.image_path), self.img_size)
        label = load_mask(item.mask_path).astype(np.float32)
        h, w = label.shape
        if (h, w) != (self.img_size, self.img_size):
            label = ndimage.zoom(label, (self.img_size / h, self.img_size / w), order=0)
        if self.augment:
            if random.random() > 0.5:
                image, label = random_rot_flip(image, label)
            elif random.random() > 0.5:
                image, label = random_rotate(image, label)
        return dict(image=torch.from_numpy(image.astype(np.float32)).unsqueeze(0),
                    label=torch.from_numpy(label.astype(np.int64)),
                    index=idx)
