"""Evaluate MedSAM2 on the test split with ground-truth box prompts.

    # fine-tuned model (output folder of medsam2/train.py)
    python medsam2/test.py --run_dir runs/medsam2_colon_20260101-120000
    # released MedSAM2 weights without fine-tuning (zero-shot)
    python medsam2/test.py --dataset colon --checkpoint medsam2/checkpoints/MedSAM2_latest.pt --out_dir runs/medsam2_zeroshot_colon

The prompt of each slice is the tight bounding box of its ground-truth tumor mask. The
low-resolution mask logits are upsampled to the original slice size and thresholded at 0.
Writes <out_dir>/{metrics.json, per_slice.csv[, predictions/]}.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tumor_segmentation/

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from common.metrics import slice_scores, summarize, write_results  # noqa: E402
from common.slices import DATASETS, DEFAULT_DATA_ROOT, load_mask  # noqa: E402
from medsam2.data import MedSAM2SliceDataset  # noqa: E402
from medsam2.model import build_medsam2, predict_logits  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", default=None, help="output folder of medsam2/train.py")
    parser.add_argument("--ckpt", default="best", help="with --run_dir: 'best' or 'last'")
    parser.add_argument("--checkpoint", default=None, help="evaluate this weight file instead of a run (e.g. MedSAM2_latest.pt)")
    parser.add_argument("--dataset", default=None, choices=DATASETS, help="required with --checkpoint")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out_dir", default=None, help="default: <run_dir>/eval_<split>_<ckpt>")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_pred", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.run_dir:
        with open(os.path.join(args.run_dir, "config.json")) as f:
            train_args = json.load(f)["args"]
        dataset, image_size = train_args["dataset"], train_args["image_size"]
        data_root = args.data_root or train_args["data_root"]
        checkpoint = os.path.join(args.run_dir, f"{args.ckpt}.pth")
        out_dir = args.out_dir or os.path.join(args.run_dir, f"eval_{args.split}_{args.ckpt}")
    else:
        if not (args.checkpoint and args.dataset and args.out_dir):
            parser.error("give --run_dir, or --checkpoint together with --dataset and --out_dir")
        dataset, image_size, checkpoint, out_dir = args.dataset, args.image_size, args.checkpoint, args.out_dir
        data_root = args.data_root or DEFAULT_DATA_ROOT
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(args.device)

    model = build_medsam2(checkpoint, device)
    model.eval()
    data = MedSAM2SliceDataset(data_root, dataset, args.split, image_size, bbox_shift=0, augment=False)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"MedSAM2 {dataset} {args.split}", ncols=90):
            low_res = predict_logits(model, batch["image"].to(device), batch["box"].to(device))
            for logits, idx in zip(low_res, batch["index"].tolist()):
                item = data.items[idx]
                gt = load_mask(item.mask_path)
                up = F.interpolate(logits[None], size=gt.shape, mode="bilinear", align_corners=False)
                pred = (up[0, 0] > 0).cpu().numpy().astype(np.uint8)
                rows.append(dict(case=item.case, slice=item.name, **slice_scores(pred, gt)))
                if args.save_pred:
                    save_dir = os.path.join(out_dir, "predictions", item.case)
                    os.makedirs(save_dir, exist_ok=True)
                    Image.fromarray(pred * 255).save(os.path.join(save_dir, item.name))

    summary = dict(model="medsam2", dataset=dataset, split=args.split, checkpoint=os.path.abspath(checkpoint),
                   prompt="ground-truth tumor bounding box", **summarize(rows))
    write_results(rows, summary, os.path.join(out_dir, "per_slice.csv"), os.path.join(out_dir, "metrics.json"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
