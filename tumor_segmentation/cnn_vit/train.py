"""Train TransUNet for 2D tumor segmentation on one dataset.

    python cnn_vit/train.py --model transunet --dataset colon

Defaults: batch 24, 150 epochs, SGD lr 0.01 (poly decay). Validation after every epoch -> best.pth (best Dice)
and last.pth.
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tumor_segmentation/

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tensorboardX import SummaryWriter  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from cnn_vit.data import SliceDataset  # noqa: E402
from cnn_vit.networks import DEFAULT_PRETRAINED_DIR, MODELS, build_model  # noqa: E402
from common.run_utils import get_logger, make_run_dir, save_config, set_seed  # noqa: E402
from common.slices import DATASETS, DEFAULT_DATA_ROOT, TUMOR_SEG_ROOT  # noqa: E402


class DiceLoss(nn.Module):
    """Soft Dice loss averaged over classes (background and tumor), as in TransUNet."""

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        loss = 0.0
        for c in range(self.n_classes):
            p, t = probs[:, c], (target == c).float()
            inter = (p * t).sum()
            loss = loss + 1 - (2 * inter + 1e-5) / ((p * p).sum() + (t * t).sum() + 1e-5)
        return loss / self.n_classes


@torch.no_grad()
def validate(model, loader, ce_loss, dice_loss, device):
    """Mean validation loss and mean slice Dice of the tumor class (at the network resolution)."""
    model.eval()
    losses, dices = [], []
    for batch in loader:
        image, label = batch["image"].to(device), batch["label"].to(device)
        logits = model(image)
        losses.append((0.5 * ce_loss(logits, label) + 0.5 * dice_loss(logits, label)).item())
        pred = logits.argmax(1)
        for p, g in zip(pred, label):
            inter = ((p == 1) & (g == 1)).sum().item()
            dices.append((2 * inter + 1e-6) / ((p == 1).sum().item() + (g == 1).sum().item() + 1e-6))
    return float(np.mean(losses)), float(np.mean(dices))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT, help="folder containing <dataset>/{train,val,test}")
    parser.add_argument("--output_dir", default=os.path.join(TUMOR_SEG_ROOT, "runs"))
    parser.add_argument("--run_name", default=None, help="default: <model>_<dataset>; a timestamp is appended")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--base_lr", type=float, default=0.01)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--deterministic", type=int, default=1, help="1: deterministic cuDNN kernels")
    parser.add_argument("--pretrained_dir", default=DEFAULT_PRETRAINED_DIR,
                        help="where the ImageNet-21k R50+ViT-B/16 weights are stored / downloaded (TransUNet)")
    parser.add_argument("--max_iters_per_epoch", type=int, default=0, help="debug: cap iterations per epoch (0 = off)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    set_seed(args.seed, bool(args.deterministic))
    run_dir = make_run_dir(args.output_dir, args.run_name or f"{args.model}_{args.dataset}")
    logger = get_logger(os.path.join(run_dir, "log.txt"))
    device = torch.device(args.device)

    train_set = SliceDataset(args.data_root, args.dataset, "train", args.img_size, augment=True)
    val_set = SliceDataset(args.data_root, args.dataset, "val", args.img_size, augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    model = build_model(args.model, num_classes=2, img_size=args.img_size, pretrained=True,
                        pretrained_dir=args.pretrained_dir).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    save_config(run_dir, args, model_parameters=n_params, train_slices=len(train_set), val_slices=len(val_set),
                train_cases=len({it.case for it in train_set.items}), val_cases=len({it.case for it in val_set.items}))
    logger.info(f"run dir: {run_dir}")
    logger.info(f"{args.model}: {n_params / 1e6:.2f}M parameters | train {len(train_set)} slices, "
                f"val {len(val_set)} slices")

    ce_loss, dice_loss = nn.CrossEntropyLoss(), DiceLoss(2)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    iters_per_epoch = len(train_loader) if not args.max_iters_per_epoch else min(len(train_loader), args.max_iters_per_epoch)
    max_iterations = args.max_epochs * iters_per_epoch
    writer = SummaryWriter(os.path.join(run_dir, "tensorboard"))
    history_path = os.path.join(run_dir, "history.csv")
    with open(history_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "lr", "train_loss", "val_loss", "val_dice", "seconds"])

    iter_num, best_dice = 0, -1.0
    for epoch in range(args.max_epochs):
        model.train()
        t0, epoch_losses = time.time(), []
        for i, batch in enumerate(train_loader):
            if args.max_iters_per_epoch and i >= args.max_iters_per_epoch:
                break
            image, label = batch["image"].to(device), batch["label"].to(device)
            logits = model(image)
            loss_ce = ce_loss(logits, label)
            loss = 0.5 * loss_ce + 0.5 * dice_loss(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr = args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9  # poly schedule
            for group in optimizer.param_groups:
                group["lr"] = lr
            iter_num += 1
            epoch_losses.append(loss.item())
            writer.add_scalar("train/loss", loss.item(), iter_num)
            writer.add_scalar("train/loss_ce", loss_ce.item(), iter_num)
            writer.add_scalar("train/lr", lr, iter_num)

        val_loss, val_dice = validate(model, val_loader, ce_loss, dice_loss, device)
        train_loss = float(np.mean(epoch_losses))
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/dice", val_dice, epoch)
        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{lr:.6g}", f"{train_loss:.5f}", f"{val_loss:.5f}", f"{val_dice:.5f}",
                                    f"{time.time() - t0:.1f}"])
        msg = f"epoch {epoch:3d}/{args.max_epochs} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | val Dice {val_dice:.4f}"
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(run_dir, "best.pth"))
            msg += "  <- best"
        logger.info(msg)

    torch.save(model.state_dict(), os.path.join(run_dir, "last.pth"))
    writer.close()
    logger.info(f"done; best val Dice {best_dice:.4f}. Test with: python cnn_vit/test.py --run_dir {run_dir}")


if __name__ == "__main__":
    main()
