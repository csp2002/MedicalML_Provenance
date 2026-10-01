"""Small helpers shared by the training/testing scripts: seeding, run folders, logging and
a record of the software environment (written to <run_dir>/config.json)."""
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def make_run_dir(output_dir, run_name):
    run_dir = os.path.join(output_dir, f"{run_name}_{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def get_logger(log_file):
    logger = logging.getLogger("run")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for h in (logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def git_commit():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=here,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def environment_info():
    info = dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), host=platform.node(), git_commit=git_commit(),
                command=" ".join(sys.argv))
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def save_config(run_dir, args, **extra):
    cfg = dict(args=vars(args), environment=environment_info(), **extra)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg
