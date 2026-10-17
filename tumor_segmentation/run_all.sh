#!/usr/bin/env bash
# Train and test the baselines on the given datasets (default: all four), one after another.
# Assumes the raw data is in data/raw/ (nnU-Net), the 2D data has been prepared (data/prepare_2d.py) and
# MedSAM2_latest.pt downloaded.
#
#   CUDA_VISIBLE_DEVICES=0 bash run_all.sh                 # 12 runs: {kits,lits,pancreas,colon} x {nnunet,transunet,medsam2}
#   CUDA_VISIBLE_DEVICES=1 bash run_all.sh colon pancreas
#   MODELS="nnunet transunet" bash run_all.sh colon        # choose the models
#   python summarize_results.py                            # table of all test results
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
DATASETS="${*:-kits lits pancreas colon}"
MODELS="${MODELS:-nnunet transunet medsam2}"

latest_run() { ls -td runs/"$1"_* | head -1; }

for d in ${DATASETS}; do
    for m in ${MODELS}; do
        case "${m}" in
            nnunet)
                bash nnunet/run_nnunet.sh "${d}" ;;
            transunet)
                python cnn_vit/train.py --model "${m}" --dataset "${d}"
                python cnn_vit/test.py --run_dir "$(latest_run "${m}_${d}")" ;;
            medsam2)
                python medsam2/train.py --dataset "${d}"
                python medsam2/test.py --run_dir "$(latest_run "medsam2_${d}")" ;;
            *) echo "unknown model ${m}" >&2; exit 1 ;;
        esac
    done
done
