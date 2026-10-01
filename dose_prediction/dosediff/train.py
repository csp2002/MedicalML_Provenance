"""Train DoseDiff on the preprocessed OpenKBP slices (multi-GPU DDP, launched with torchrun).

    cd dose_prediction/dosediff
    torchrun --nproc_per_node=4 train.py --bs 16          # 4 GPUs x 16 = effective batch 64
    torchrun --nproc_per_node=1 train.py --bs 16          # single GPU

Recipe: T = 1000 linear-beta DDPM, epsilon prediction, MSE loss, AdamW (lr 2e-4, weight
decay 1e-4), lr x0.1 at 70% of training, gradient clipping 1.0, 1600 epochs.
Every `--val_every` epochs (and after the first and the last epoch) the model samples the validation slices
with 4-step DDIM and the MAE (Gy) inside the possible-dose mask is logged; model_epoch<N>.pth is
saved and the lowest-MAE model is kept as model_best_mae.pth.
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid

from dataset import OpenKBPSliceDataset
from guided_diffusion.resample import create_named_schedule_sampler
from model import BODY_CHANNEL, IMG_SIZE, build_diffusion, build_model

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default=os.path.join(HERE, "..", "data", "preprocessed", "NPY"),
                        help="folder with train/ and validation/ written by preprocess/nii_to_npy.py")
    parser.add_argument("--output_dir", default=os.path.join(HERE, "..", "runs"))
    parser.add_argument("--bs", type=int, default=16, help="batch size per GPU")
    parser.add_argument("--T", type=int, default=1000, help="number of diffusion steps")
    parser.add_argument("--epochs", type=int, default=1600)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_every", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_iters_per_epoch", type=int, default=0, help="debug: cap iterations per epoch (0 = off)")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    train_data = OpenKBPSliceDataset(os.path.join(args.data_dir, "train"), augment=True)
    val_data = OpenKBPSliceDataset(os.path.join(args.data_dir, "validation"), augment=False)
    train_sampler = DistributedSampler(train_data, shuffle=True, seed=args.seed)
    val_sampler = DistributedSampler(val_data, shuffle=False)
    train_loader = DataLoader(train_data, batch_size=args.bs, sampler=train_sampler, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=2 * args.bs, sampler=val_sampler, num_workers=args.num_workers,
                            pin_memory=True)

    model = DDP(build_model().cuda(), device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False,
                bucket_cap_mb=128, find_unused_parameters=False)
    diffusion = build_diffusion(args.T)
    diffusion_val = build_diffusion(args.T, "ddim4")
    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = MultiStepLR(optimizer, milestones=[int(0.7 * args.epochs)], gamma=0.1)

    run_dir, writer = None, None
    if rank == 0:
        effective_bs = world_size * args.bs
        run_dir = os.path.join(args.output_dir, f"dosediff_T{args.T}_bs{effective_bs}_ep{args.epochs}_"
                                                f"{time.strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(run_dir)
        n_params = sum(p.numel() for p in model.parameters())
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(dict(args=vars(args), world_size=world_size, effective_batch_size=effective_bs,
                           model_parameters=n_params, train_slices=len(train_data), val_slices=len(val_data),
                           environment=dict(python=platform.python_version(), torch=torch.__version__,
                                            cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(local_rank),
                                            host=platform.node(), command=" ".join(sys.argv))), f, indent=2)
        writer = SummaryWriter(os.path.join(run_dir, "tensorboard"))
        with open(os.path.join(run_dir, "val_history.csv"), "w") as f:
            f.write("epoch,val_mae_gy\n")
        print(f"run dir: {run_dir}\n{n_params / 1e6:.1f}M parameters | train {len(train_data)} slices, "
              f"val {len(val_data)} slices | {world_size} GPU(s) x batch {args.bs}", flush=True)

    best_mae = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        lr = optimizer.param_groups[0]["lr"]
        losses, t0 = [], time.time()
        for i, (ct, cond, dose) in enumerate(train_loader):
            if args.max_iters_per_epoch and i >= args.max_iters_per_epoch:
                break
            ct, cond, dose = ct.cuda().float(), cond.cuda().float(), dose.cuda().float()
            t, weights = schedule_sampler.sample(dose.shape[0], dose.device)
            out = diffusion.training_losses(model=model, x_start=dose, t=t, model_kwargs={"ct": ct, "dis": cond})
            loss = (out["loss"] * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            losses.append(loss.item() / world_size)
        lr_scheduler.step()
        if rank == 0:
            writer.add_scalar("lr", lr, epoch + 1)
            writer.add_scalar("train_loss", np.mean(losses), epoch + 1)
            print(f"epoch {epoch + 1}/{args.epochs} | lr {lr:.2e} | train loss {np.mean(losses):.4f} | "
                  f"{time.time() - t0:.0f}s", flush=True)

        if epoch == 0 or (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
            model.eval()
            maes, shown = [], []
            with torch.no_grad():
                for i, (ct, cond, dose) in enumerate(val_loader):
                    ct, cond, dose = ct.cuda().float(), cond.cuda().float(), dose.cuda().float()
                    pred = diffusion_val.ddim_sample_loop(model, (ct.size(0), 1, *IMG_SIZE), clip_denoised=True,
                                                          model_kwargs={"ct": ct, "dis": cond}, eta=0.0)
                    dose_gy, pred_gy = (dose + 1) * 40, (pred + 1) * 40
                    body = cond[:, BODY_CHANNEL:BODY_CHANNEL + 1]
                    # As in the original DoseDiff code: MAE per batch (and rank), averaged over ranks and
                    # batches -- a close approximation of the voxel-weighted MAE, used only for model selection.
                    mae = (torch.abs(dose_gy - pred_gy) * body).sum() / body.sum()
                    dist.all_reduce(mae, op=dist.ReduceOp.SUM)
                    maes.append(mae.item() / world_size)
                    if i in (0, 2, 4, 8):
                        shown.append((ct[:1].cpu(), dose_gy[:1].cpu(), pred_gy[:1].cpu()))
            if rank == 0:
                val_mae = float(np.mean(maes))
                writer.add_scalar("val_MAE", val_mae, epoch + 1)
                for j, tag in enumerate(["image_CT", "true_dose", "pred_dose"]):
                    writer.add_image(tag, make_grid(torch.cat([s[j] for s in shown]), 2, normalize=True), epoch + 1)
                state = model.module.state_dict()
                torch.save(state, os.path.join(run_dir, f"model_epoch{epoch + 1}.pth"))
                msg = f"epoch {epoch + 1}: val MAE {val_mae:.4f} Gy"
                if val_mae < best_mae:
                    best_mae = val_mae
                    torch.save(state, os.path.join(run_dir, "model_best_mae.pth"))
                    msg += "  <- best"
                print(msg, flush=True)
                with open(os.path.join(run_dir, "val_history.csv"), "a") as f:
                    f.write(f"{epoch + 1},{val_mae:.5f}\n")

    if rank == 0:
        writer.close()
        print(f"done; best val MAE {best_mae:.4f} Gy. Test with: python test.py --run_dir {run_dir}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
