#!/bin/bash
set -e

source /usr/local/miniforge3/etc/profile.d/conda.sh

NNUNET_RAW=/home/ge.polymtl.ca/p123239/data/nnUNetv2/nnUNet_raw
NNUNET_PREP=/home/ge.polymtl.ca/p123239/data/nnUNetv2/nnUNet_preprocessed
NNUNET_RES=/home/ge.polymtl.ca/p123239/data/nnUNetv2/nnUNet_results
IMAGES_TS=$NNUNET_RAW/Dataset026_BrnoSpineAll/imagesTs
LABELS_TS=$NNUNET_RAW/Dataset026_BrnoSpineAll/labelsTs
DATASET_JSON=$NNUNET_RAW/Dataset026_BrnoSpineAll/dataset.json
LOG_DIR=/home/ge.polymtl.ca/p123239/SpineFoundation/logs
GPU=1

mkdir -p $LOG_DIR

export nnUNet_raw=$NNUNET_RAW
export nnUNet_preprocessed=$NNUNET_PREP
export nnUNet_results=$NNUNET_RES
export CUDA_VISIBLE_DEVICES=$GPU
export nnUNet_n_proc_DA=8
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export PYTHONUNBUFFERED=1

for fold in 0 1 2 3 4 5 6 7 8; do
    echo "=========================================="
    echo "FOLD $fold"
    echo "=========================================="

    # ── MedDINOv3 ──────────────────────────────────────────────────────────────
    PRED_MED=$NNUNET_RES/predictions_meddinov3_fold${fold}
    CKPT_MED=$NNUNET_RES/Dataset026_BrnoSpineAll/meddinov3_base_primus_multiscale_Trainer__nnUNetPlans__2d/fold_${fold}/checkpoint_latest.pth

    echo "[fold $fold] Training MedDINOv3..."
    conda activate dinov3
    if [ -f "$CKPT_MED" ]; then
        nnUNetv2_train 26 2d $fold -tr meddinov3_base_primus_multiscale_Trainer --c \
            2>&1 | tee $LOG_DIR/train_meddinov3_fold${fold}.log
    else
        nnUNetv2_train 26 2d $fold -tr meddinov3_base_primus_multiscale_Trainer \
            2>&1 | tee $LOG_DIR/train_meddinov3_fold${fold}.log
    fi

    echo "[fold $fold] Inference MedDINOv3..."
    nnUNetv2_predict \
        -i $IMAGES_TS \
        -o $PRED_MED \
        -d 26 -c 2d -f $fold \
        -tr meddinov3_base_primus_multiscale_Trainer \
        -chk checkpoint_best.pth \
        2>&1 | tee $LOG_DIR/predict_meddinov3_fold${fold}.log

    echo "[fold $fold] Evaluation MedDINOv3..."
    conda activate dino
    nnUNetv2_evaluate_folder \
        $LABELS_TS $PRED_MED \
        -djfile $DATASET_JSON \
        -pfile $PRED_MED/plans.json \
        2>&1 | tee $LOG_DIR/eval_meddinov3_fold${fold}.log

    # ── nnUNetTrainer ──────────────────────────────────────────────────────────
    PRED_NN=$NNUNET_RES/predictions_nnunet_fold${fold}
    CKPT_NN=$NNUNET_RES/Dataset026_BrnoSpineAll/nnUNetTrainer__nnUNetPlans__2d/fold_${fold}/checkpoint_latest.pth

    echo "[fold $fold] Training nnUNetTrainer..."
    if [ -f "$CKPT_NN" ]; then
        nnUNetv2_train 26 2d $fold --c \
            2>&1 | tee $LOG_DIR/train_nnunet_fold${fold}.log
    else
        nnUNetv2_train 26 2d $fold \
            2>&1 | tee $LOG_DIR/train_nnunet_fold${fold}.log
    fi

    echo "[fold $fold] Inference nnUNetTrainer..."
    nnUNetv2_predict \
        -i $IMAGES_TS \
        -o $PRED_NN \
        -d 26 -c 2d -f $fold \
        -chk checkpoint_best.pth \
        2>&1 | tee $LOG_DIR/predict_nnunet_fold${fold}.log

    echo "[fold $fold] Evaluation nnUNetTrainer..."
    nnUNetv2_evaluate_folder \
        $LABELS_TS $PRED_NN \
        -djfile $DATASET_JSON \
        -pfile $PRED_NN/plans.json \
        2>&1 | tee $LOG_DIR/eval_nnunet_fold${fold}.log

    echo "[fold $fold] Done."
done

echo "=========================================="
echo "Toutes les expériences terminées."
echo "=========================================="

python3 - <<'EOF'
import json
from pathlib import Path

RES = Path("/home/ge.polymtl.ca/p123239/data/nnUNetv2/nnUNet_results")
FOLDS = range(9)
SUBJECTS = [1, 2, 3, 4, 5, 10, 20, 30, 50]

print(f"{'Fold':>5} {'Subjects':>8} {'nnUNet Dice':>12} {'MedDINOv3 Dice':>15}")
print("-" * 45)
for fold, n_subj in zip(FOLDS, SUBJECTS):
    def get_dice(name):
        p = RES / f"predictions_{name}_fold{fold}" / "summary.json"
        if p.exists():
            return f"{json.load(open(p))['foreground_mean']['Dice']:.4f}"
        return "N/A"
    print(f"{fold:>5} {n_subj:>8} {get_dice('nnunet'):>12} {get_dice('meddinov3'):>15}")
EOF
