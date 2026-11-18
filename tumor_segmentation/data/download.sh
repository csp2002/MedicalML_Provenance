#!/usr/bin/env bash
# Download the raw 3D datasets into tumor_segmentation/data/raw/<dataset folder> (other parent folder: RAW_DIR=...).
#
#   bash data/download.sh colon      # MSD Task10_Colon    (~6.2 GB tar)
#   bash data/download.sh pancreas   # MSD Task07_Pancreas (~12.3 GB tar)
#   bash data/download.sh kits       # KiTS23 (git clone + official download script, ~50 GB)
#   bash data/download.sh lits       # prints manual instructions (registration required)
#
# After downloading, run data/prepare_2d.py (see tumor_segmentation/README.md).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW="${RAW_DIR:-$HERE/raw}"
mkdir -p "$RAW"

msd() {
    # MSD archives from the AWS mirror of the MONAI project (the same files as on http://medicaldecathlon.com/)
    local task="$1" md5="$2"
    local url="https://msd-for-monai.s3-us-west-2.amazonaws.com/${task}.tar"
    echo "Downloading ${url} -> ${RAW}/${task}.tar"
    wget -c -O "${RAW}/${task}.tar" "${url}"
    echo "Checking md5"
    echo "${md5}  ${RAW}/${task}.tar" | md5sum -c -
    echo "Extracting ${task}.tar"
    tar -xf "${RAW}/${task}.tar" -C "${RAW}"
    echo "Done: ${RAW}/${task} (you may delete ${RAW}/${task}.tar)"
}

case "${1:-}" in
    colon)
        msd Task10_Colon bad7a188931dc2f6acf72b08eb6202d0
        ;;
    pancreas)
        msd Task07_Pancreas 4f7080cfca169fa8066d17ce6eb061e4
        ;;
    kits)
        # Official KiTS23 repository at the tested commit; its download module fetches the CT volumes (Hugging Face)
        # into kits23/dataset/case_XXXXX/imaging.nii.gz. An existing kits23 folder is reused.
        if [ ! -d "${RAW}/kits23" ]; then
            git clone https://github.com/neheller/kits23 "${RAW}/kits23"
            git -C "${RAW}/kits23" checkout -q c108835
        fi
        (cd "${RAW}/kits23" && python -m kits23.download)
        ;;
    lits)
        cat <<EOF
LiTS (Liver Tumor Segmentation Challenge, 2017) requires registration, so it cannot be
downloaded automatically. Get the two training archives ("Training Batch 1" and
"Training Batch 2", 131 CT volumes in total) from the official challenge page
    https://competitions.codalab.org/competitions/17094
and unpack all of their files into ONE flat folder:
    ${RAW}/LITS17/volume-0.nii ... volume-130.nii
    ${RAW}/LITS17/segmentation-0.nii ... segmentation-130.nii
(~50 GB unpacked). Only the 118 volumes that contain liver tumor are used (see splits/lits.json).
EOF
        ;;
    *)
        echo "usage: bash data/download.sh {colon|pancreas|kits|lits}" >&2
        exit 1
        ;;
esac
