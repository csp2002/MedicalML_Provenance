"""Fine-tune MedSAM2 with LoRA on the 2D tumor slices of one dataset (box prompt = ground-truth tumor box).

    python medsam2/train.py --dataset colon

Recipe: start from MedSAM2_latest.pt; LoRA (rank 4, alpha 4) on q/v of the image encoder, prompt encoder and mask
decoder trained; AdamW, lr 1e-3 (LoRA) / 5e-5 (decoder); loss 20 * focal + Dice; batch 8, 70 epochs.
Validation after every epoch -> best.pth (best Dice) and last.pth, with the LoRA weights merged in
(same format as MedSAM2_latest.pt).
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tumor_segmentation/

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tensorboardX import SummaryWriter  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from common.run_utils import get_logger, make_run_dir, save_config, set_seed  # noqa: E402
from common.slices import DATASETS, DEFAULT_DATA_ROOT, TUMOR_SEG_ROOT  # noqa: E402
from medsam2.data import MedSAM2SliceDataset  # noqa: E402
from medsam2.lora import apply_lora, is_lora_param, merged_state_dict  # noqa: E402
from medsam2.model import DEFAULT_CHECKPOINT, build_medsam2, predict_logits  # noqa: E402


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Per-sample focal loss averaged over pixels (as in SAM2's training code)."""
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * (1 - p_t) ** gamma
    loss = (alpha * targets + (1 - alpha) * (1 - targets)) * loss
    return loss.flatten(1).mean(1)


def dice_loss(logits, targets):
    prob = logits.sigmoid().flatten(1)
    targets = targets.flatten(1)
    return 1 - (2 * (prob * targets).sum(1) + 1) / (prob.sum(1) + targets.sum(1) + 1)


def forward_loss(model, batch, device, image_size):
    images, masks, boxes = batch["image"].to(device), batch["mask"].to(device), batch["box"].to(device)
    logits = F.interpolate(predict_logits(model, images, boxes), size=(image_size, image_size),
                           mode="bilinear", align_corners=False)
    focal, dice = sigmoid_focal_loss(logits, masks), dice_loss(logits, masks)
    return (20.0 * focal + dice).mean(), focal.mean(), dice.mean(), logits


@torch.no_grad()
def validate(model, loader, device, image_size):
    model.eval()
    losses, dices = [], []
    for batch in loader:
        loss, _, _, logits = forward_loss(model, batch, device, image_size)
        losses.append(loss.item())
        pred = logits > 0
        gt = batch["mask"].to(device) > 0.5
        inter = (pred & gt).flatten(1).sum(1).float()
        dices += ((2 * inter + 1e-6) / (pred.flatten(1).sum(1) + gt.flatten(1).sum(1) + 1e-6)).tolist()
    return float(np.mean(losses)), float(np.mean(dices))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="pretrained MedSAM2 weights")
    parser.add_argument("--output_dir", default=os.path.join(TUMOR_SEG_ROOT, "runs"))
    parser.add_argument("--run_name", default=None, help="default: medsam2_<dataset>; a timestamp is appended")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--lr", type=float, default=5e-5, help="lr of prompt encoder + mask decoder")
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=4.0)
    parser.add_argument("--lora_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--bbox_shift", type=int, default=0,
                        help="max random box enlargement (pixels) in training; 0 = the same tight GT boxes as at test "
                             "time (enlarged training boxes lowered the test Dice with tight test boxes)")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--max_iters_per_epoch", type=int, default=0, help="debug: cap iterations per epoch (0 = off)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.lora_r < 1:
        parser.error("--lora_r must be >= 1")

    set_seed(args.seed, deterministic=False)
    run_dir = make_run_dir(args.output_dir, args.run_name or f"medsam2_{args.dataset}")
    logger = get_logger(os.path.join(run_dir, "log.txt"))
    device = torch.device(args.device)

    model = build_medsam2(args.checkpoint, device)
    for p in model.parameters():
        p.requires_grad = False
    n_blocks = apply_lora(model.image_encoder, r=args.lora_r, alpha=args.lora_alpha)
    for module in (model.sam_prompt_encoder, model.sam_mask_decoder):
        for p in module.parameters():
            p.requires_grad = True
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and is_lora_param(n)]
    other_params = [p for n, p in model.named_parameters() if p.requires_grad and not is_lora_param(n)]
    optimizer = torch.optim.AdamW([{"params": lora_params, "lr": args.lora_lr},
                                   {"params": other_params, "lr": args.lr}],
                                  betas=(0.9, 0.999), weight_decay=args.weight_decay)
    n_total = sum(p.numel() for p in model.parameters())
    n_lora, n_other = sum(p.numel() for p in lora_params), sum(p.numel() for p in other_params)

    train_set = MedSAM2SliceDataset(args.data_root, args.dataset, "train", args.image_size,
                                    bbox_shift=args.bbox_shift, augment=True)
    val_set = MedSAM2SliceDataset(args.data_root, args.dataset, "val", args.image_size, bbox_shift=0, augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
    save_config(run_dir, args, model_parameters=n_total, trainable_lora=n_lora, trainable_decoder=n_other,
                lora_blocks=n_blocks, train_slices=len(train_set), val_slices=len(val_set),
                train_cases=len({it.case for it in train_set.items}), val_cases=len({it.case for it in val_set.items}))
    logger.info(f"run dir: {run_dir}")
    logger.info(f"MedSAM2 {n_total / 1e6:.2f}M parameters; trainable: LoRA {n_lora / 1e6:.3f}M on {n_blocks} "
                f"Hiera blocks + prompt encoder/mask decoder {n_other / 1e6:.2f}M")
    logger.info(f"train {len(train_set)} slices, val {len(val_set)} slices")

    writer = SummaryWriter(os.path.join(run_dir, "tensorboard"))
    history_path = os.path.join(run_dir, "history.csv")
    with open(history_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_focal", "train_dice_loss", "val_loss", "val_dice", "seconds"])

    best_dice, step = -1.0, 0
    for epoch in range(args.epochs):
        model.train()
        t0, stats = time.time(), []
        for i, batch in enumerate(train_loader):
            if args.max_iters_per_epoch and i >= args.max_iters_per_epoch:
                break
            loss, focal, dice, _ = forward_loss(model, batch, device, args.image_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            stats.append((loss.item(), focal.item(), dice.item()))
            step += 1
            writer.add_scalar("train/loss", loss.item(), step)
        train_loss, train_focal, train_dice = np.mean(stats, axis=0)
        val_loss, val_dice = validate(model, val_loader, device, args.image_size)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/dice", val_dice, epoch)
        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.5f}", f"{train_focal:.5f}", f"{train_dice:.5f}",
                                    f"{val_loss:.5f}", f"{val_dice:.5f}", f"{time.time() - t0:.1f}"])
        msg = f"epoch {epoch:3d}/{args.epochs} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | val Dice {val_dice:.4f}"
        ckpt = {"model": merged_state_dict(model), "epoch": epoch, "val_dice": val_dice}
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(ckpt, os.path.join(run_dir, "best.pth"))
            msg += "  <- best"
        logger.info(msg)

    torch.save(ckpt, os.path.join(run_dir, "last.pth"))
    writer.close()
    logger.info(f"done; best val Dice {best_dice:.4f}. Test with: python medsam2/test.py --run_dir {run_dir}")


if __name__ == "__main__":
    main()
