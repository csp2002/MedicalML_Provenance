# MedML-Baselines: tumor segmentation and radiotherapy dose prediction

Data preparation, training and testing code for standard baseline models on two medical imaging tasks, all on
public data:

| Task | Datasets | Baseline models | Metrics |
|---|---|---|---|
| [**Tumor segmentation**](tumor_segmentation/README.md) (tumor-bearing CT slices) | KiTS23 (kidney), LiTS17 (liver), MSD Pancreas, MSD Colon | nnU-Net (CNN), TransUNet (ViT), MedSAM2 (foundation model, LoRA fine-tuning) | Dice, IoU |
| [**Dose prediction**](dose_prediction/README.md) (3D, head & neck IMRT) | OpenKBP | DoseDiff (conditional diffusion model), C3D (cascade 3D U-Net) | OpenKBP dose score, DVH score |

This gives 12 segmentation training cases (4 datasets × 3 models) and 2 dose-prediction training cases.

## Repository layout

```
.
├── requirements.txt
├── tumor_segmentation/
│   ├── data/            download.sh, compute_hu_windows.py, prepare_2d.py, splits/{kits,lits,pancreas,colon}.json
│   ├── common/          slice listing, metrics, run bookkeeping
│   ├── nnunet/          nnU-Net on tumor-only 3D sub-volumes (prepare_nnunet.py, run_nnunet.sh, evaluate.py)
│   ├── cnn_vit/         TransUNet (train.py, test.py)
│   └── medsam2/         MedSAM2 with LoRA (train.py, test.py; vendored sam2 package)
└── dose_prediction/
    ├── data/            download.sh (OpenKBP)
    ├── preprocess/      csv_to_nii.py -> make_psdm.py -> nii_to_npy.py
    ├── common/          OpenKBP evaluation (evaluate_openkbp.py)
    ├── dosediff/        DoseDiff (train.py, test.py; vendored guided_diffusion)
    └── c3d/             C3D (train.py, test.py)
```

## Installation

Tested on Linux with Python 3.10, PyTorch 2.5.1 + CUDA 12.4 and NVIDIA RTX A6000 GPUs:

```bash
conda create -n medml python=3.10 -y
conda activate medml
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Use the PyTorch wheel that matches your CUDA driver. The scripts are run directly; nothing else needs to be installed.

## Quick start

**Tumor segmentation** (colon as example; replace with `kits`, `lits` or `pancreas`):

```bash
cd tumor_segmentation
bash data/download.sh colon                                   # raw data -> data/raw/Task10_Colon
python data/prepare_2d.py --dataset colon                     # 2D slices -> data/2D/colon
bash nnunet/run_nnunet.sh colon                               # nnU-Net: prepare, train, predict, evaluate -> runs/nnunet_colon_nnUNetTrainer/
python cnn_vit/train.py --model transunet --dataset colon     # -> runs/transunet_colon_<timestamp>/
bash medsam2/checkpoints/download.sh                          # MedSAM2_latest.pt
python medsam2/train.py --dataset colon                       # -> runs/medsam2_colon_<timestamp>/
python cnn_vit/test.py --run_dir runs/transunet_colon_<timestamp>
python medsam2/test.py --run_dir runs/medsam2_colon_<timestamp>
```

**Dose prediction:**

```bash
cd dose_prediction
bash data/download.sh                                         # OpenKBP -> data/open-kbp
python preprocess/csv_to_nii.py && python preprocess/make_psdm.py && python preprocess/nii_to_npy.py
cd dosediff
torchrun --nproc_per_node=4 train.py --bs 16                  # 4 GPUs -> ../runs/dosediff_T1000_bs64_ep1600_<timestamp>/
python test.py --run_dir ../runs/dosediff_T1000_bs64_ep1600_<timestamp> --tta
cd ../c3d                                                     # C3D needs only preprocess/csv_to_nii.py
python train.py                                               # -> ../runs/c3d_bs2_it80000_<timestamp>/
python test.py --run_dir ../runs/c3d_bs2_it80000_<timestamp> --tta
```

## References

Datasets
* KiTS23 — N. Heller et al., "The KiTS21 Challenge: Automatic segmentation of kidneys, renal tumors, and renal cysts in corticomedullary-phase CT", arXiv:2307.01984, 2023.
* LiTS — P. Bilic et al., "The Liver Tumor Segmentation Benchmark (LiTS)", *Medical Image Analysis* 84:102680, 2023.
* MSD — M. Antonelli et al., "The Medical Segmentation Decathlon", *Nature Communications* 13:4128, 2022; A. L. Simpson et al., arXiv:1902.09063, 2019.
* OpenKBP — A. Babier et al., "OpenKBP: The open-access knowledge-based planning grand challenge and dataset", *Medical Physics* 48:5549–5561, 2021.

Models
* nnU-Net — F. Isensee et al., "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation", *Nature Methods* 18:203–211, 2021.
* TransUNet — J. Chen et al., "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation", arXiv:2102.04306, 2021.
* MedSAM2 — J. Ma et al., "MedSAM2: Segment Anything in 3D Medical Images and Videos", arXiv:2504.03600, 2025; SAM 2 — N. Ravi et al., ICLR 2025.
* LoRA — E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022.
* DoseDiff — Y. Zhang et al., "DoseDiff: Distance-Aware Diffusion Model for Dose Prediction in Radiotherapy", *IEEE TMI* 43(10):3621–3633, 2024.
* C3D — S. Liu et al., "Technical Note: A cascade 3D U-Net for dose prediction in radiotherapy", *Medical Physics* 48(9):5574–5582, 2021.

<!--
## Licenses

This repository is released under the Apache-2.0 license ([LICENSE](LICENSE)). It includes code from:

| Component | Source | License | Location |
|---|---|---|---|
| TransUNet | [Beckschen/TransUNet](https://github.com/Beckschen/TransUNet) | Apache-2.0 | `tumor_segmentation/cnn_vit/networks/transunet/` |
| SAM2 / MedSAM2 (`sam2` package) | [bowang-lab/MedSAM2](https://github.com/bowang-lab/MedSAM2) | Apache-2.0 | `tumor_segmentation/medsam2/sam2/` |
| DoseDiff | [whisney/DoseDiff](https://github.com/whisney/DoseDiff) | MIT | `dose_prediction/dosediff/`, `dose_prediction/preprocess/` |
| guided-diffusion | [openai/guided-diffusion](https://github.com/openai/guided-diffusion) | MIT | `dose_prediction/dosediff/guided_diffusion/` |
| OpenKBP evaluation | [ababier/open-kbp](https://github.com/ababier/open-kbp) | MIT | `dose_prediction/common/evaluate_openkbp.py` |
| C3D | [RTDosePrediction](https://github.com/yangsenwxy/RTDosePrediction) | — | `dose_prediction/c3d/` |

The license texts are in the respective folders. nnU-Net is installed from PyPI. The datasets remain under their
own terms (KiTS23: CC BY-NC-SA; MSD: CC-BY-SA 4.0; LiTS and OpenKBP: see the challenge pages).
-->
