"""Slice dataset for MedSAM2 (box-prompted 2D segmentation).

Each 8-bit CT slice is replicated to 3 channels, resized to 512x512 (cubic) and
normalised with the ImageNet mean/std used by SAM2. The box prompt is the tight
bounding box of the ground-truth tumor mask of the slice (one box around all tumor
pixels); with bbox_shift > 0 every side is enlarged by a random 0-`bbox_shift` pixels.
Training augmentation: horizontal flip (p=0.5), random affine (rotation +-25 deg,
shear +-20 deg), colour jitter (brightness 0.1, contrast 0.03, saturation 0.03) and
random grayscale (p=0.05).
"""
import random

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from skimage import transform
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode

from common.slices import list_slices, load_image, load_mask

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def mask_to_box(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


class MedSAM2SliceDataset(Dataset):
    def __init__(self, data_root, dataset, split, image_size=512, bbox_shift=0, augment=False):
        self.items = list_slices(data_root, dataset, split)
        self.image_size = image_size
        self.bbox_shift = bbox_shift
        self.augment = augment
        self.color_jitter = T.ColorJitter(brightness=0.1, contrast=0.03, saturation=0.03, hue=0)

    def __len__(self):
        return len(self.items)

    def _load(self, item):
        s = self.image_size
        image = load_image(item.image_path)
        image = np.repeat(image[:, :, None], 3, axis=-1)
        image = transform.resize(image, (s, s), order=3, preserve_range=True, anti_aliasing=True)
        image = np.clip(image, 0, 255).astype(np.uint8)
        mask_orig = load_mask(item.mask_path)
        mask = transform.resize(mask_orig.astype(np.float32), (s, s), order=0, preserve_range=True,
                                anti_aliasing=False)
        # box of the original-resolution mask in input-pixel units; used when a tiny tumor
        # vanishes in the nearest-neighbour resize
        h, w = mask_orig.shape
        fallback_box = mask_to_box(mask_orig) * np.array([s / w, s / h, s / w, s / h], dtype=np.float32)
        return Image.fromarray(image), Image.fromarray((mask > 0.5).astype(np.uint8) * 255), fallback_box

    def _augment(self, img, mask):
        if random.random() < 0.5:
            img, mask = TF.hflip(img), TF.hflip(mask)
        angle, translate, scale, shear = T.RandomAffine.get_params(
            degrees=[-25, 25], translate=None, scale_ranges=None, shears=[-20, 20], img_size=[self.image_size] * 2)
        img = TF.affine(img, angle, translate, scale, shear, interpolation=InterpolationMode.BILINEAR, fill=0)
        mask = TF.affine(mask, angle, translate, scale, shear, interpolation=InterpolationMode.NEAREST, fill=0)
        img = self.color_jitter(img)
        if random.random() < 0.05:
            img = TF.rgb_to_grayscale(img, num_output_channels=3)
        return img, mask

    def __getitem__(self, idx):
        item = self.items[idx]
        img, mask, box = self._load(item)
        if self.augment:
            aug_img, aug_mask = self._augment(img, mask)
            if np.asarray(aug_mask).max() > 0:  # keep the un-augmented slice if the tumor left the frame
                img, mask = aug_img, aug_mask
        mask = (np.asarray(mask) > 127).astype(np.float32)
        if mask.any():
            box = mask_to_box(mask)
        if self.bbox_shift > 0:
            s = self.image_size
            box[0] = max(0, box[0] - random.randint(0, self.bbox_shift))
            box[1] = max(0, box[1] - random.randint(0, self.bbox_shift))
            box[2] = min(s - 1, box[2] + random.randint(0, self.bbox_shift))
            box[3] = min(s - 1, box[3] + random.randint(0, self.bbox_shift))
        image = TF.normalize(TF.to_tensor(img), IMAGENET_MEAN, IMAGENET_STD)
        return dict(image=image, mask=torch.from_numpy(mask).unsqueeze(0), box=torch.from_numpy(box), index=idx)
