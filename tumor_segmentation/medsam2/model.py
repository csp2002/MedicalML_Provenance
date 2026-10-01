"""Building MedSAM2 and running it on a batch of 2D slices with box prompts.

`predict_logits` reproduces the single-image path of SAM2's `SAM2ImagePredictor`
(image embedding + `no_mem_embed`, box given as two corner points with labels 2/3,
single-mask output) in a batched, differentiable form, so the exact same forward pass
is used for fine-tuning and for inference.
"""
import os
import sys

import torch

MEDSAM2_DIR = os.path.dirname(os.path.abspath(__file__))
if MEDSAM2_DIR not in sys.path:
    sys.path.insert(0, MEDSAM2_DIR)  # makes the vendored `sam2` package importable

from sam2.build_sam import build_sam2  # noqa: E402

CONFIG = "configs/sam2.1_hiera_t512.yaml"  # MedSAM2 = SAM2.1 Hiera-Tiny at 512x512
DEFAULT_CHECKPOINT = os.path.join(MEDSAM2_DIR, "checkpoints", "MedSAM2_latest.pt")


def build_medsam2(checkpoint, device):
    """MedSAM2 with the weights of `checkpoint` (the released MedSAM2_latest.pt or a fine-tuned
    checkpoint written by medsam2/train.py -- both store the weights under the key 'model')."""
    return build_sam2(CONFIG, ckpt_path=checkpoint, device=device, mode="eval", apply_postprocessing=False)


def predict_logits(model, images, boxes):
    """images: (B, 3, S, S) normalised; boxes: (B, 4) [x0, y0, x1, y1] in input-pixel units.
    Returns low-resolution mask logits of shape (B, 1, S/4, S/4)."""
    b = images.shape[0]
    backbone_out = model.forward_image(images)
    _, vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out)
    if model.directly_add_no_mem_embed:
        vision_feats[-1] = vision_feats[-1] + model.no_mem_embed
    feats = [f.permute(1, 2, 0).reshape(b, -1, *size) for f, size in zip(vision_feats, feat_sizes)]
    image_embed, high_res_feats = feats[-1], feats[:-1]

    coords = boxes.reshape(b, 2, 2).float()
    labels = torch.tensor([[2, 3]], dtype=torch.int, device=images.device).repeat(b, 1)
    sparse, dense = model.sam_prompt_encoder(points=(coords, labels), boxes=None, masks=None)
    low_res_masks, _, _, _ = model.sam_mask_decoder(
        image_embeddings=image_embed,
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res_feats,
    )
    return low_res_masks
