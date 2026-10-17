import copy
import os
import urllib.request

import numpy as np

from .transunet.vit_seg_modeling import CONFIGS, VisionTransformer

MODELS = ("transunet",)

# ImageNet-21k pre-trained R50+ViT-B/16 released by Google (used to initialise the TransUNet encoder).
R50_VIT_B16_URL = "https://storage.googleapis.com/vit_models/imagenet21k/R50%2BViT-B_16.npz"
DEFAULT_PRETRAINED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "pretrained")


def _fetch(url, path):
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading {url} -> {path}")
        urllib.request.urlretrieve(url, path + ".partial")
        os.replace(path + ".partial", path)
    return path


def build_model(name, num_classes=2, img_size=224, pretrained=True, pretrained_dir=DEFAULT_PRETRAINED_DIR):
    """TransUNet, initialised with the ImageNet-21k R50+ViT-B/16 weights."""
    if name == "transunet":
        config = copy.deepcopy(CONFIGS["R50-ViT-B_16"])
        config.n_classes = num_classes
        config.n_skip = 3
        config.patches.grid = (img_size // 16, img_size // 16)
        net = VisionTransformer(config, img_size=img_size, num_classes=num_classes)
        if pretrained:
            weights = _fetch(R50_VIT_B16_URL, os.path.join(pretrained_dir, "R50+ViT-B_16.npz"))
            net.load_from(weights=np.load(weights))
        return net
    raise ValueError(f"unknown model {name!r}; choose from {MODELS}")
