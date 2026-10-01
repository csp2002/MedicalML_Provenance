#!/usr/bin/env bash
# nnU-Net (v2, 3d_fullres) on the tumor-only sub-volumes of one dataset: prepare -> plan/preprocess -> train fold 0
# (= this repository's train / val split) -> predict the test cases -> score them like the 2D baselines.
#
#   bash nnunet/run_nnunet.sh colon                     # run from tumor_segmentation/
#   RAW_DIR=/path/to/raw bash nnunet/run_nnunet.sh colon   # raw data in /path/to/raw/Task10_Colon (as data/download.sh)
#   TRAINER=nnUNetTrainer_5epochs bash nnunet/run_nnunet.sh colon   # quick functional test
#   DEVICE=cpu TRAINER=nnUNetTrainer_1epoch bash nnunet/run_nnunet.sh colon
#
# nnU-Net folders default to data/nnUNet_raw, data/nnUNet_preprocessed and runs/nnUNet_results (override with
# the usual nnUNet_raw / nnUNet_preprocessed / nnUNet_results environment variables).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATASET="${1:?usage: bash nnunet/run_nnunet.sh kits|lits|pancreas|colon}"
declare -A ID=( [kits]=511 [lits]=512 [pancreas]=513 [colon]=514 )
declare -A NAME=( [kits]=Dataset511_KiTSTO [lits]=Dataset512_LiTSTO [pancreas]=Dataset513_PancreasTO [colon]=Dataset514_ColonTO )
declare -A RAWNAME=( [kits]=kits23 [lits]=LITS17 [pancreas]=Task07_Pancreas [colon]=Task10_Colon )
DSID="${ID[$DATASET]}"; DSNAME="${NAME[$DATASET]}"
TRAINER="${TRAINER:-nnUNetTrainer}"
DEVICE="${DEVICE:-cuda}"
export nnUNet_raw="${nnUNet_raw:-$PWD/data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$PWD/data/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$PWD/runs/nnUNet_results}"
RUN_DIR="${RUN_DIR:-runs/nnunet_${DATASET}_${TRAINER}}"

# 1. tumor-only sub-volumes in nnU-Net format (skipped if already there)
if [ ! -f "$nnUNet_raw/$DSNAME/dataset.json" ]; then
    python nnunet/prepare_nnunet.py --dataset "$DATASET" ${RAW_DIR:+--raw_dir "$RAW_DIR/${RAWNAME[$DATASET]}"}
fi
# 2. experiment planning + preprocessing, then fold 0 := our train / val split
nnUNetv2_plan_and_preprocess -d "$DSID" -c 3d_fullres --verify_dataset_integrity
cp "$nnUNet_raw/$DSNAME/splits_final.json" "$nnUNet_preprocessed/$DSNAME/splits_final.json"
# 3. training (1000 epochs x 250 iterations by default). To resume an interrupted run, export the three nnUNet_*
#    folders above and run: nnUNetv2_train $DSID 3d_fullres 0 -tr $TRAINER --c
nnUNetv2_train "$DSID" 3d_fullres 0 -tr "$TRAINER" -device "$DEVICE"
# 4. prediction of the test cases with the final checkpoint
mkdir -p "$RUN_DIR"
nnUNetv2_predict -i "$nnUNet_raw/$DSNAME/imagesTs" -o "$RUN_DIR/pred_test" -d "$DSID" -c 3d_fullres -f 0 -tr "$TRAINER" -device "$DEVICE"
# 5. slice-level Dice / IoU on the tumor slices (same metric as the 2D baselines)
python nnunet/evaluate.py --dataset "$DATASET" --pred_dir "$RUN_DIR/pred_test" --out_dir "$RUN_DIR/eval_test" \
    --checkpoint "$nnUNet_results/$DSNAME/${TRAINER}__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
