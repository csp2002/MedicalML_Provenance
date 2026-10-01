"""DoseDiff network, diffusion schedule and conditioning channels (shared by train.py and test.py).

DoseDiff (Zhang et al., IEEE TMI 2024; https://github.com/whisney/DoseDiff) is a conditional
DDPM that denoises a 2D dose slice given the CT slice (1 channel) and a 20-channel
"distance" condition:
    [0]     fused PTV mask = PTV70 + 63/70 * PTV63 + 56/70 * PTV56
    [1-8]   masks of Brainstem, Esophagus, Larynx, LeftParotid, Mandible, possible_dose_mask,
            RightParotid, SpinalCord                (index 6 = possible_dose_mask = body)
    [9-19]  signed distance maps (PSDM) of Brainstem, Esophagus, Larynx, LeftParotid, Mandible,
            possible_dose_mask, PTV56, PTV63, PTV70, RightParotid, SpinalCord
"""
import numpy as np
import torch

from guided_diffusion import gaussian_diffusion as gd
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.unet import UNetModel_MS_Former

IMG_SIZE = (128, 128)
MASK_CHANNELS = ["Mask_Brainstem", "Mask_Esophagus", "Mask_Larynx", "Mask_LeftParotid", "Mask_Mandible",
                 "Mask_possible_dose_mask", "Mask_RightParotid", "Mask_SpinalCord"]
PSDM_CHANNELS = ["PSDM_Brainstem", "PSDM_Esophagus", "PSDM_Larynx", "PSDM_LeftParotid", "PSDM_Mandible",
                 "PSDM_possible_dose_mask", "PSDM_PTV56", "PSDM_PTV63", "PSDM_PTV70", "PSDM_RightParotid",
                 "PSDM_SpinalCord"]
PTV_WEIGHTS = {"Mask_PTV70": 70.0 / 70.0, "Mask_PTV63": 63.0 / 70.0, "Mask_PTV56": 56.0 / 70.0}
BODY_CHANNEL = 6  # index of Mask_possible_dose_mask in the 20-channel condition
DIS_CHANNELS = 1 + len(MASK_CHANNELS) + len(PSDM_CHANNELS)  # 20


def build_condition(fields):
    """fields: dict name -> array (..., H, W). Returns the 20 condition channels stacked on axis 0."""
    ptv = sum(w * fields[name] for name, w in PTV_WEIGHTS.items())
    return np.stack([ptv] + [fields[n] for n in MASK_CHANNELS] + [fields[n] for n in PSDM_CHANNELS], axis=0)


def build_model():
    return UNetModel_MS_Former(image_size=IMG_SIZE, in_channels=1, ct_channels=1, dis_channels=DIS_CHANNELS,
                               model_channels=128, out_channels=1, num_res_blocks=2, attention_resolutions=(16, 32),
                               dropout=0, channel_mult=(1, 1, 2, 3, 4), conv_resample=True, dims=2, num_classes=None,
                               use_checkpoint=False, use_fp16=False, num_heads=4, num_head_channels=-1,
                               num_heads_upsample=-1, use_scale_shift_norm=True, resblock_updown=False,
                               use_new_attention_order=False)


def build_diffusion(T=1000, steps=None):
    """steps=None: the full T-step DDPM (training); steps='ddim8' etc.: a respaced DDIM sampler."""
    return SpacedDiffusion(use_timesteps=space_timesteps(T, [T] if steps is None else steps),
                           betas=gd.get_named_beta_schedule("linear", T),
                           model_mean_type=gd.ModelMeanType.EPSILON,
                           model_var_type=gd.ModelVarType.FIXED_LARGE,
                           loss_type=gd.LossType.MSE, rescale_timesteps=False)


def load_weights(model, path):
    """Loads a checkpoint written by train.py (also accepts DDP-prefixed 'module.' keys)."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict({k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state.items()})
    return model
