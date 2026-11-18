# Radiotherapy dose prediction (OpenKBP: DoseDiff, C3D)

Predict the 3D dose distribution (Gy) of a head-and-neck IMRT plan from the planning CT and the contours of the
targets and organs at risk, with two baselines:

| Model | Family | Input | Code |
|---|---|---|---|
| **DoseDiff** (Zhang et al., IEEE TMI 2024) | conditional diffusion model, 2D | one axial slice: CT + masks and signed distance maps of the structures | [`dosediff/`](dosediff) |
| **C3D** (Liu et al., Medical Physics 2021; 1st place in the OpenKBP challenge) | cascade of two 3D U-Nets | the whole 128³ volume: CT, PTVs and organ-at-risk masks | [`c3d/`](c3d) |

All commands are run from this folder (`dose_prediction/`) unless a `cd` is shown.

```
dose_prediction/
├── data/download.sh             # clone the OpenKBP repository (data in CSV format)
├── preprocess/
│   ├── csv_to_nii.py            # step 1: sparse CSV -> 128^3 NIfTI
│   ├── make_psdm.py             # step 2: + per-structure masks and signed distance maps (PSDM)
│   ├── nii_to_npy.py            # step 3: -> normalised 2D slices for training
│   └── nii_utils.py
├── common/
│   └── evaluate_openkbp.py      # OpenKBP dose score and DVH score (used by both models)
├── dosediff/
│   ├── guided_diffusion/        # diffusion code + UNetModel_MS_Former (vendored from DoseDiff / OpenAI guided-diffusion)
│   ├── model.py                 # network / schedule / conditioning channels
│   └── dataset.py, train.py, test.py
└── c3d/                         # C3D, copied from the C3D authors' RTDosePrediction
    ├── model.py, loss.py, online_evaluation.py, train.py, test.py
    └── DataLoader/, DataAugmentation/, NetworkTrainer/
```

## 1. Dataset: OpenKBP

[OpenKBP](https://github.com/ababier/open-kbp): 340 head-and-neck cancer patients (official split 200 train /
40 validation / 100 test), each with a 128³ CT, the clinical dose, the mask of voxels that may receive dose and the
contours of up to 7 organs at risk and 3 targets (PTV56, PTV63, PTV70).

```bash
bash data/download.sh      # git clone https://github.com/ababier/open-kbp -> data/open-kbp (2 GB)
```

To clone elsewhere, set `OPENKBP_DIR=...` and pass `--provided_data $OPENKBP_DIR/provided-data` to `csv_to_nii.py`.

## 2. Preprocessing

```bash
python preprocess/csv_to_nii.py      # step 1 (both models): -> data/preprocessed/{train,validation,test}-pats/pt_*/*.nii.gz
python preprocess/make_psdm.py       # step 2 (DoseDiff): masks and signed distance maps -> data/preprocessed/<split>-pats_preprocess/
python preprocess/nii_to_npy.py      # step 3 (DoseDiff): 2D training slices -> data/preprocessed/NPY/ (≈31 GB)
```

Step 1 writes 340 patient folders (200 / 40 / 100); step 3 writes 16,456 training and 3,294 validation slices.

## 3. Training

### 3.1 DoseDiff

```bash
cd dosediff
torchrun --nproc_per_node=4 train.py --bs 16      # 4 GPUs x 16 = effective batch 64
```

`--nproc_per_node` is the number of GPUs; the effective batch (`bs<batch>` in the run-folder name) is GPUs × `--bs`.

| Setting | Value |
|---|---|
| Network | `UNetModel_MS_Former` (179 M parameters) |
| Diffusion | T = 1000, linear β schedule, ε-prediction, MSE loss |
| Augmentation | shift / scale / rotate, horizontal and vertical flips |
| Optimiser | AdamW, lr 2e-4, weight decay 1e-4, lr × 0.1 after 70 % of the epochs |
| Batch / epochs | 64 (4 GPUs × 16) / 1600 |
| Validation | every 10 epochs (4-step DDIM, MAE in Gy) → `model_best_mae.pth` |

Each run writes `runs/dosediff_T1000_bs<batch>_ep<epochs>_<timestamp>/` (`config.json`, `val_history.csv`,
TensorBoard logs, checkpoints).

### 3.2 C3D

```bash
cd c3d
python train.py                                     # batch 2, 80,000 iterations, GPU 0
python train.py --batch_size 4 --list_GPU_ids 0 1   # batch 4 on two GPUs
```

| Setting | Value |
|---|---|
| Network | two cascaded 3D U-Nets (32 M parameters); the second takes the features of the first and the input |
| Loss | 0.5 × L1 (first U-Net) + L1 (second U-Net), inside the possible-dose mask |
| Augmentation | flips, rotation about the slice axis, random shift |
| Optimiser | Adam (AMSGrad), lr 3e-4, weight decay 3e-5, cosine decay |
| Batch / iterations | 2 / 80,000 |
| Validation | every 500 iterations (dose score of the 40 validation patients) → `best_val_evaluation_index.pkl` |

Each run writes `runs/c3d_bs<batch>_it<iterations>_<timestamp>/` (`config.json`, `log.txt` with the validation dose
scores, `best_val_evaluation_index.pkl`, `latest.pkl`); `python train.py --resume <run_dir>/latest.pkl` (with the
run's other arguments) continues an interrupted run.

## 4. Testing

```bash
cd dosediff
python test.py --run_dir ../runs/dosediff_T1000_bs64_ep1600_<timestamp> --tta    # model_best_mae.pth, test split
cd ../c3d
python test.py --run_dir ../runs/c3d_bs2_it80000_<timestamp> --tta              # best_val_evaluation_index.pkl, test split
```

DoseDiff samples each axial slice with 8-step DDIM; C3D predicts the whole volume; `--tta` averages over 4 flips.
The predicted dose (Gy, zero outside the possible-dose mask) is scored with the OpenKBP metrics:

* **Dose score**: mean absolute dose error (Gy) inside the possible-dose mask, averaged over patients.
* **DVH score**: mean absolute error (Gy) of the DVH metrics D1, D95, D99 (targets) and D0.1cc, Dmean (OARs).

Results go to an `eval_<split>_<ckpt>...` folder inside the run folder (`predictions/`, `metrics.json`,
`per_patient.csv`). The trained C3D model published by its authors
([`C3D_bs4_iter80000.pkl`](https://drive.google.com/file/d/1YaXkOCfH-nrKD3nli5_Z6stF1rYWMA-v/view)) can be tested
directly: `python test.py --ckpt /path/to/C3D_bs4_iter80000.pkl --out_dir ../runs/eval_c3d_published --tta`.

## 5. Expected results and compute

Test split (100 patients) with TTA; lower is better:

| | Dose score (Gy) | DVH score (Gy) | Training | Testing |
|---|---|---|---|---|
| DoseDiff | ≈ 2.6–2.7 | ≈ 1.5–1.6 | ≈ 60 h on 4 × RTX A6000 | ≈ 2 h on one RTX A6000 |
| C3D (the authors' published model) | ≈ 2.5 | ≈ 1.5 | ≈ 18 GB GPU memory at batch 2; ≈ 50 h on two GTX 1080 Ti (reported by the C3D authors) | ≈ 17 min on CPU (`--GPU_id -1`, 32 threads) |

The best validation MAE of DoseDiff in `val_history.csv` is about 2.5 Gy.
