#!/usr/bin/env bash
# Download the released MedSAM2 weights (SAM2.1 Hiera-T fine-tuned on medical data, ~150 MB)
# from the official Hugging Face repository (https://huggingface.co/wanglab/MedSAM2).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
wget -c -O "${HERE}/MedSAM2_latest.pt" "https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt"
echo "saved ${HERE}/MedSAM2_latest.pt"
