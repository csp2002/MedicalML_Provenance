"""Predict the dose of every test patient with a trained DoseDiff model and score it with the OpenKBP metrics.

    cd dose_prediction/dosediff
    python test.py --run_dir ../runs/dosediff_T1000_bs64_ep1600_20260101-120000 --tta
    python test.py --ckpt /path/to/model_best_mae.pth --out_dir ../runs/eval_x --split validation

Each axial slice is sampled with DDIM (8 steps by default); --tta averages over 4 flips. The dose (Gy) is zeroed
outside the possible-dose mask. Writes <out_dir>/{predictions/pt_*/dose.nii.gz, metrics.json, per_patient.csv}.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "preprocess"))
sys.path.insert(0, os.path.join(HERE, "..", "common"))

from evaluate_openkbp import evaluate  # noqa: E402
from model import IMG_SIZE, MASK_CHANNELS, PSDM_CHANNELS, PTV_WEIGHTS, build_condition, build_diffusion, build_model, load_weights  # noqa: E402
from nii_utils import nii_read, nii_write  # noqa: E402

FLIPS = [None, [2], [3], [2, 3]]  # TTA: identity, vertical, horizontal, both


def load_patient(case_dir):
    ct, spacing, origin, direction = nii_read(os.path.join(case_dir, "ct.nii.gz"))
    fields = {k: nii_read(os.path.join(case_dir, f"{k}.nii.gz"))[0]
              for k in list(PTV_WEIGHTS) + MASK_CHANNELS + PSDM_CHANNELS}
    ct = np.clip(ct, 0, 2500) / 1250 - 1
    cond = build_condition(fields)  # (20, D, H, W)
    return ct, cond, fields["Mask_possible_dose_mask"], (spacing, origin, direction)


@torch.no_grad()
def predict_volume(model, diffusion, ct, cond, bs, tta):
    pred = np.zeros(ct.shape, dtype=np.float64)
    flips = FLIPS if tta else FLIPS[:1]
    for start in range(0, ct.shape[0], bs):
        ct_b = torch.from_numpy(ct[start:start + bs]).unsqueeze(1).float()             # (b, 1, H, W)
        cond_b = torch.from_numpy(cond[:, start:start + bs]).permute(1, 0, 2, 3).float()  # (b, 20, H, W)
        for dims in flips:
            x_ct = (ct_b if dims is None else torch.flip(ct_b, dims=dims)).cuda()
            x_cond = (cond_b if dims is None else torch.flip(cond_b, dims=dims)).cuda()
            out = diffusion.ddim_sample_loop(model, (x_ct.size(0), 1, *IMG_SIZE), clip_denoised=True, eta=0.0,
                                             model_kwargs={"ct": x_ct, "dis": x_cond})
            if dims is not None:
                out = torch.flip(out, dims=dims)
            pred[start:start + bs] += out[:, 0].cpu().numpy()
    return pred / len(flips)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", default=None, help="output folder of train.py")
    parser.add_argument("--ckpt", default="model_best_mae.pth", help="file name inside --run_dir, or a full path")
    parser.add_argument("--data_dir", default=os.path.join(HERE, "..", "data", "preprocessed"))
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--out_dir", default=None, help="default: <run_dir>/eval_<split>_<ckpt>_ddim<N>[_tta]")
    parser.add_argument("--T", type=int, default=1000, help="must match training")
    parser.add_argument("--ddim", type=int, default=8, help="number of DDIM sampling steps")
    parser.add_argument("--tta", action="store_true", help="average over 4 flips")
    parser.add_argument("--bs", type=int, default=64, help="slices per sampling batch")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ckpt = args.ckpt if os.path.isabs(args.ckpt) or args.run_dir is None else os.path.join(args.run_dir, args.ckpt)
    tag = f"eval_{args.split}_{os.path.splitext(os.path.basename(ckpt))[0]}_ddim{args.ddim}{'_tta' if args.tta else ''}"
    out_dir = args.out_dir or os.path.join(args.run_dir, tag)
    os.makedirs(os.path.join(out_dir, "predictions"), exist_ok=True)
    torch.manual_seed(args.seed)

    model = load_weights(build_model(), ckpt).cuda().eval()
    diffusion = build_diffusion(args.T, f"ddim{args.ddim}")
    case_root = os.path.join(args.data_dir, f"{args.split}-pats_preprocess")
    patients = sorted(os.listdir(case_root))
    t0 = time.time()
    for i, pt in enumerate(patients, 1):
        ct, cond, body, geometry = load_patient(os.path.join(case_root, pt))
        pred = predict_volume(model, diffusion, ct, cond, args.bs, args.tta)
        pred = np.clip((pred + 1) * 40, 0, 80) * body
        os.makedirs(os.path.join(out_dir, "predictions", pt), exist_ok=True)
        nii_write(os.path.join(out_dir, "predictions", pt, "dose.nii.gz"), pred, *geometry)
        print(f"[{i}/{len(patients)}] {pt} ({time.time() - t0:.0f}s)", flush=True)

    summary, rows = evaluate(os.path.join(out_dir, "predictions"), os.path.join(args.data_dir, f"{args.split}-pats"))
    summary.update(checkpoint=os.path.abspath(ckpt), split=args.split, ddim_steps=args.ddim, tta=args.tta,
                   seed=args.seed)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "per_patient.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient", "dose_mae", "dvh_mae"])
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
