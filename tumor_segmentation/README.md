# Tumor segmentation (2D CT slices)

Binary tumor segmentation on the tumor-bearing axial CT slices of four public datasets:

| Model | Family | Initialisation | Input | Code |
|---|---|---|---|---|
| **nnU-Net** | CNN (self-configuring U-Net, `3d_fullres`) | random | tumor-only 3D sub-volume | [`nnunet/`](nnunet) |
| **TransUNet** | ViT (R50 + ViT-B/16 hybrid encoder) | ImageNet-21k | 1 × 224 × 224 slice | [`cnn_vit/`](cnn_vit) |
| **MedSAM2** | foundation model (SAM2.1 Hiera-T), box-prompted | MedSAM2 weights, LoRA fine-tuning | 3 × 512 × 512 slice + box | [`medsam2/`](medsam2) |

Models are trained and evaluated only on the axial slices that contain tumor, and all models are evaluated on the
same test slices. nnU-Net works on the 3D stack of tumor slices of a case, the other models on single slices.
All commands are run from this folder (`tumor_segmentation/`).

```
tumor_segmentation/
├── data/
│   ├── download.sh           # raw data download (MSD / KiTS23) or instructions (LiTS)
│   ├── compute_hu_windows.py # HU windows of prepare_2d.py from the training cases
│   ├── prepare_2d.py         # raw 3D NIfTI -> 2D PNG slice dataset
│   └── splits/*.json         # fixed case-level train / val / test splits
├── common/                   # slice listing, metrics, run bookkeeping (shared)
├── nnunet/                   # nnU-Net: prepare_nnunet.py, run_nnunet.sh, evaluate.py
├── cnn_vit/                  # TransUNet: networks/, data.py, train.py, test.py
├── medsam2/                  # MedSAM2: sam2/ (vendored), lora.py, model.py, data.py, train.py, test.py
├── run_all.sh                # train + test every model on every dataset
└── summarize_results.py      # collect all metrics.json into one table
```

## 1. Datasets

| Dataset | Tumor | Source | Label used | Cases (train / val / test) | Tumor slices (train / val / test) |
|---|---|---|---|---|---|
| `kits` | kidney tumor | [KiTS23](https://github.com/neheller/kits23) | 2 (tumor) | 342 / 48 / 99 | 8263 / 1380 / 2539 |
| `lits` | liver tumor | [LiTS17](https://competitions.codalab.org/competitions/17094) | 2 (tumor) | 83 / 11 / 24 | 4694 / 681 / 1815 |
| `pancreas` | pancreatic tumor | [MSD](http://medicaldecathlon.com/) Task07_Pancreas | 2 (cancer) | 196 / 28 / 57 | 1732 / 223 / 582 |
| `colon` | colon cancer | [MSD](http://medicaldecathlon.com/) Task10_Colon | 1 (cancer) | 89 / 11 / 26 | 939 / 92 / 254 |

Only the labelled (training) cases of each challenge are used, split at the case level with the fixed splits in
[`data/splits/`](data/splits). The 13 LiTS volumes without tumor are not used.

### 1.1 Download

```bash
bash data/download.sh colon      # MSD Task10_Colon    -> data/raw/Task10_Colon     (6.2 GB, md5-checked)
bash data/download.sh pancreas   # MSD Task07_Pancreas -> data/raw/Task07_Pancreas  (12.3 GB, md5-checked)
bash data/download.sh kits       # KiTS23 repo + its download module -> data/raw/kits23 (~50 GB)
bash data/download.sh lits       # prints the manual steps (registration required)
```

LiTS requires registration: download "Training Batch 1" and "Training Batch 2" from the
[challenge page](https://competitions.codalab.org/competitions/17094) and put all `volume-*.nii` and
`segmentation-*.nii` files in `data/raw/LITS17/`. To use another location, set `RAW_DIR=/other/dir` for
`download.sh` and `run_nnunet.sh`, and pass `--raw_dir /other/dir/<dataset folder>` to the preprocessing scripts.

Expected raw layout:

```
data/raw/kits23/dataset/case_00000/{imaging,segmentation}.nii.gz
data/raw/LITS17/{volume,segmentation}-<N>.nii
data/raw/Task07_Pancreas/{imagesTr,labelsTr}/pancreas_<NNN>.nii.gz
data/raw/Task10_Colon/{imagesTr,labelsTr}/colon_<NNN>.nii.gz
```

### 1.2 2D slices (TransUNet, MedSAM2)

```bash
python data/prepare_2d.py --dataset colon                     # reads data/raw/Task10_Colon
python data/prepare_2d.py --dataset kits --raw_dir /path/to/kits23 --num_workers 8
```

The script keeps the axial slices that contain tumor, clips the CT to the HU window of the dataset, scales it to
[0, 1] and writes each slice and its binary tumor mask as PNG files. The windows are the 0.5 / 99.5 percentiles of
the tumor voxels of the training cases (computed by `data/compute_hu_windows.py`):

| Dataset | kits | lits | pancreas | colon |
|---|---|---|---|---|
| HU window | [-45, 227] | [-52, 167] | [-37, 205] | [-31, 171] |

Output (kits 1.8 GB, lits 1.4 GB, pancreas 0.4 GB, colon 0.2 GB):

```
data/2D/<dataset>/
├── {train,val,test}/CT/<case>/<slice>.png      # CT slice
├── {train,val,test}/Mask/<case>/<slice>.png    # tumor mask (0 / 255)
├── {train,val,test}_slices.csv                 # one row per slice, with its source volume
└── prepare_config.json                         # parameters + sha256 of the split file
```

### 1.3 nnU-Net data

```bash
python nnunet/prepare_nnunet.py --dataset colon       # -> data/nnUNet_raw/Dataset514_ColonTO
```

Each case is cropped along the axial axis to the range from its first to its last tumor slice, and the label is
binarised. The script writes the nnU-Net raw dataset (train + val cases in `imagesTr`, test cases in `imagesTs`)
and `splits_final.json`, whose fold 0 is the train / val split above. Dataset ids: 511 kits, 512 lits,
513 pancreas, 514 colon.

## 2. Training

### 2.1 nnU-Net

```bash
bash nnunet/run_nnunet.sh colon          # prepare (if needed) -> plan/preprocess -> train fold 0 -> predict test -> evaluate
```

The script runs the standard nnU-Net v2 commands with nnU-Net's default recipe (1000 epochs). Step by step:

```bash
export nnUNet_raw=$PWD/data/nnUNet_raw nnUNet_preprocessed=$PWD/data/nnUNet_preprocessed nnUNet_results=$PWD/runs/nnUNet_results
nnUNetv2_plan_and_preprocess -d 514 -c 3d_fullres --verify_dataset_integrity
cp data/nnUNet_raw/Dataset514_ColonTO/splits_final.json data/nnUNet_preprocessed/Dataset514_ColonTO/
nnUNetv2_train 514 3d_fullres 0
nnUNetv2_predict -i data/nnUNet_raw/Dataset514_ColonTO/imagesTs -o runs/nnunet_colon_nnUNetTrainer/pred_test -d 514 -c 3d_fullres -f 0
python nnunet/evaluate.py --dataset colon --pred_dir runs/nnunet_colon_nnUNetTrainer/pred_test --out_dir runs/nnunet_colon_nnUNetTrainer/eval_test
```

An interrupted training is resumed with `nnUNetv2_train 514 3d_fullres 0 --c`.
`TRAINER=nnUNetTrainer_5epochs bash nnunet/run_nnunet.sh colon` is a quick functional test.

### 2.2 TransUNet

```bash
python cnn_vit/train.py --model transunet --dataset colon    # downloads R50+ViT-B_16.npz (ImageNet-21k) on first use
```

| Setting | Value |
|---|---|
| Input | CT slice resized to 224 × 224 |
| Augmentation | random 90° rotation + flip, random rotation in [-20°, 20°) |
| Loss | 0.5 × cross-entropy + 0.5 × Dice |
| Optimiser | SGD (momentum 0.9, weight decay 1e-4), lr 0.01 with poly decay |
| Batch / epochs | 24 / 150 |
| Checkpoints | `best.pth` (best validation Dice), `last.pth` |

### 2.3 MedSAM2

```bash
bash medsam2/checkpoints/download.sh      # MedSAM2_latest.pt from https://huggingface.co/wanglab/MedSAM2
python medsam2/train.py --dataset colon
```

| Setting | Value |
|---|---|
| Backbone | MedSAM2 (SAM2.1 Hiera-Tiny, 512 × 512) |
| Prompt | bounding box of the tumor in the slice (from the ground truth), in training and testing |
| Fine-tuning | LoRA (rank 4, alpha 4) on q/v of the image encoder; prompt encoder and mask decoder fully trained |
| Augmentation | flip, random affine, colour jitter |
| Loss | 20 × focal + Dice |
| Optimiser | AdamW: lr 1e-3 (LoRA), 5e-5 (prompt encoder, mask decoder) |
| Batch / epochs | 8 / 70 |
| Checkpoints | `best.pth` (best validation Dice), `last.pth` |

### 2.4 Run folders

Each run of `cnn_vit/` and `medsam2/` writes `runs/<model>_<dataset>_<timestamp>/`:

```
config.json      # arguments, data sizes, environment, git commit
log.txt, history.csv, tensorboard/
best.pth, last.pth
```

## 3. Testing

```bash
python nnunet/evaluate.py --dataset colon --pred_dir runs/nnunet_colon_nnUNetTrainer/pred_test \
    --out_dir runs/nnunet_colon_nnUNetTrainer/eval_test                      # (run_nnunet.sh does this)
python cnn_vit/test.py --run_dir runs/transunet_colon_<timestamp>              # --ckpt best|last, --save_pred
python medsam2/test.py --run_dir runs/medsam2_colon_<timestamp>
# MedSAM2 without fine-tuning (zero-shot, same box prompts):
python medsam2/test.py --dataset colon --checkpoint medsam2/checkpoints/MedSAM2_latest.pt --out_dir runs/medsam2_zeroshot_colon
```

Predictions are compared with the ground truth on every tumor slice of the test cases at the original resolution.
Reported: per-slice Dice and IoU averaged over all test slices (`slice_dice_mean`, `slice_iou_mean`) and Dice / IoU
pooled per case (`case_dice_mean`, `case_iou_mean`); nnU-Net also reports the 3D Dice of the sub-volume. Results go
to `<run_dir>/eval_<split>_<ckpt>/` (`metrics.json`, `per_slice.csv`, predicted masks with `--save_pred`).

### All 12 cases in one go

Needs the raw data (nnU-Net), the 2D data and `MedSAM2_latest.pt`:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_all.sh                  # {kits,lits,pancreas,colon} x {nnunet,transunet,medsam2}: train + test
CUDA_VISIBLE_DEVICES=1 bash run_all.sh colon pancreas   # a subset of the datasets
python summarize_results.py --csv results.csv           # one table with the test metrics of every run under runs/
```

## 4. Expected results

Approximate test slice Dice, to check that a run worked (single runs differ by a few points):

| | kits | lits | pancreas | colon |
|---|---|---|---|---|
| nnU-Net | ≈ 0.83 | ≈ 0.70 | ≈ 0.57 | ≈ 0.66 |
| TransUNet | ≈ 0.55 | ≈ 0.53 | ≈ 0.24 | ≈ 0.35–0.43 |
| MedSAM2, LoRA fine-tuned | – | – | – | ≈ 0.88–0.89 |
| MedSAM2, released weights (zero-shot) | ≈ 0.91 | ≈ 0.71 | ≈ 0.91 | ≈ 0.85 |

* MedSAM2 gets the ground-truth box of each slice as prompt, so its values are not comparable with those of the
  fully automatic models. Fine-tuning should not end below the zero-shot values.

## 5. Compute

Measured on shared NVIDIA RTX A6000 GPUs:

| | GPU memory (default batch) | Training time |
|---|---|---|
| nnU-Net (3d_fullres) | ~8 GB | 50–80 s per epoch, 1000 epochs: ~14 h (colon) to ~22 h (pancreas) per dataset |
| TransUNet | ~10 GB (24) | ~16 s per epoch on colon, 150 epochs ≈ 0.7 h |
| MedSAM2 (LoRA) | ~5 GB (8) | ~70 s per epoch on colon, 70 epochs ≈ 1.4 h |

For the slice models the epoch time grows with the number of training slices (KiTS has 8.8× as many as colon:
roughly 6–12 h per model).
