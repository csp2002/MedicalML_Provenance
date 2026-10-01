"""Evaluate a trained UNet++ / TransUNet run on the test split.

    python cnn_vit/test.py --run_dir runs/transunet_colon_20260101-120000
    python cnn_vit/test.py --run_dir <run_dir> --ckpt last --save_pred

Predictions are made at the network resolution (224x224), resized back to the original
slice size (nearest neighbour) and compared with the original mask. Writes
<run_dir>/eval_<split>_<ckpt>/{metrics.json, per_slice.csv[, predictions/]}.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tumor_segmentation/

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from cnn_vit.data import SliceDataset  # noqa: E402
from cnn_vit.networks import build_model  # noqa: E402
from common.metrics import slice_scores, summarize, write_results  # noqa: E402
from common.slices import load_mask  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", required=True, help="output folder of cnn_vit/train.py")
    parser.add_argument("--ckpt", default="best", help="'best', 'last' or a path to a .pth file")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data_root", default=None, help="default: the data_root used for training")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_pred", action="store_true", help="also write predicted masks as PNG")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(os.path.join(args.run_dir, "config.json")) as f:
        train_args = json.load(f)["args"]
    data_root = args.data_root or train_args["data_root"]
    ckpt_path = args.ckpt if args.ckpt.endswith(".pth") else os.path.join(args.run_dir, f"{args.ckpt}.pth")
    ckpt_tag = os.path.splitext(os.path.basename(ckpt_path))[0]
    out_dir = os.path.join(args.run_dir, f"eval_{args.split}_{ckpt_tag}")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(args.device)

    model = build_model(train_args["model"], num_classes=2, img_size=train_args["img_size"], pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.to(device).eval()

    dataset = SliceDataset(data_root, train_args["dataset"], args.split, train_args["img_size"], augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{train_args['model']} {train_args['dataset']} {args.split}", ncols=90):
            preds = model(batch["image"].to(device)).argmax(1).cpu().numpy().astype(np.uint8)
            for pred, idx in zip(preds, batch["index"].tolist()):
                item = dataset.items[idx]
                gt = load_mask(item.mask_path)
                if pred.shape != gt.shape:
                    pred = ndimage.zoom(pred, (gt.shape[0] / pred.shape[0], gt.shape[1] / pred.shape[1]), order=0)
                rows.append(dict(case=item.case, slice=item.name, **slice_scores(pred, gt)))
                if args.save_pred:
                    save_dir = os.path.join(out_dir, "predictions", item.case)
                    os.makedirs(save_dir, exist_ok=True)
                    Image.fromarray(pred * 255).save(os.path.join(save_dir, item.name))

    summary = dict(model=train_args["model"], dataset=train_args["dataset"], split=args.split,
                   checkpoint=os.path.abspath(ckpt_path), **summarize(rows))
    write_results(rows, summary, os.path.join(out_dir, "per_slice.csv"), os.path.join(out_dir, "metrics.json"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
