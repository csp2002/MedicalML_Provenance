# -*- encoding: utf-8 -*-
# Copied from RTDosePrediction (https://github.com/yangsenwxy/RTDosePrediction), Src/C3D/train.py.
# Changed: data / output folders as arguments, a time-stamped run folder with config.json, --resume.
"""Train C3D on the volumes of preprocess/csv_to_nii.py.

    cd dose_prediction/c3d
    python train.py                                       # batch 2, 80,000 iterations, GPU 0
    python train.py --batch_size 4 --list_GPU_ids 0 1     # batch 4 on two GPUs
    python train.py --resume ../runs/c3d_bs2_it80000_<timestamp>/latest.pkl
"""
import argparse
import json
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))

import torch  # noqa: E402

from DataLoader.dataloader_OpenKBP_C3D import get_loader  # noqa: E402
from NetworkTrainer.network_trainer import NetworkTrainer  # noqa: E402
from model import Model  # noqa: E402
from online_evaluation import online_evaluation  # noqa: E402
from loss import Loss  # noqa: E402

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2,
                        help='batch size for training (default: 2)')
    parser.add_argument('--list_GPU_ids', nargs='+', type=int, default=[0],
                        help='list_GPU_ids for training (default: [0]; -1: CPU)')
    parser.add_argument('--max_iter',  type=int, default=80000,
                        help='training iterations(default: 80000)')
    parser.add_argument('--data_dir', default=os.path.join(HERE, '..', 'data', 'preprocessed'),
                        help='folder with train-pats/ and validation-pats/ written by preprocess/csv_to_nii.py')
    parser.add_argument('--output_dir', default=os.path.join(HERE, '..', 'runs'))
    parser.add_argument('--resume', default=None, help='latest.pkl of an interrupted run (continues in its folder)')
    args = parser.parse_args()

    #  Start training
    trainer = NetworkTrainer()
    trainer.setting.project_name = 'C3D'
    if args.resume:
        trainer.setting.output_dir = os.path.dirname(os.path.abspath(args.resume))
    else:
        trainer.setting.output_dir = os.path.join(args.output_dir, 'c3d_bs%d_it%d_%s' % (
            args.batch_size, args.max_iter, time.strftime('%Y%m%d-%H%M%S')))
    list_GPU_ids = args.list_GPU_ids

    trainer.setting.network = Model(in_ch=9, out_ch=1,
                                    list_ch_A=[-1, 16, 32, 64, 128, 256],
                                    list_ch_B=[-1, 32, 64, 128, 256, 512])

    trainer.setting.max_iter = args.max_iter

    trainer.setting.train_loader, trainer.setting.val_loader = get_loader(
        data_dir=args.data_dir,
        train_bs=args.batch_size,
        val_bs=1,
        train_num_samples_per_epoch=args.batch_size * 500,  # 500 iterations per epoch
        val_num_samples_per_epoch=1,
        num_works=4
    )
    val_dir = os.path.join(args.data_dir, 'validation-pats')
    trainer.setting.val_patient_dirs = [os.path.join(val_dir, p) for p in sorted(os.listdir(val_dir)) if p.startswith('pt_')]

    trainer.setting.eps_train_loss = 0.01
    trainer.setting.lr_scheduler_update_on_iter = True
    trainer.setting.loss_function = Loss()
    trainer.setting.online_evaluation_function_val = online_evaluation

    trainer.set_optimizer(optimizer_type='Adam',
                          args={
                              'lr': 3e-4,
                              'weight_decay': 1e-4
                          }
                          )

    trainer.set_lr_scheduler(lr_scheduler_type='cosine',
                             args={
                                 'T_max': args.max_iter,
                                 'eta_min': 1e-7,
                                 'last_epoch': -1
                             }
                             )

    if not os.path.exists(trainer.setting.output_dir):
        os.makedirs(trainer.setting.output_dir)
    if args.resume:
        trainer.init_trainer(ckpt_file=args.resume, list_GPU_ids=list_GPU_ids, only_network=False)
    else:
        with open(os.path.join(trainer.setting.output_dir, 'config.json'), 'w') as f:
            json.dump(dict(args=vars(args), environment=dict(
                python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                host=platform.node(), command=' '.join(sys.argv))), f, indent=2)
        trainer.set_GPU_device(list_GPU_ids)
    trainer.run()

    trainer.print_log_to_file('# Done !\n', 'a')
