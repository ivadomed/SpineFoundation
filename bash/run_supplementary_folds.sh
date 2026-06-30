#!/bin/bash
# Génère et entraîne les folds supplémentaires 6, 7, 8, 9 sujets (indices 9–12).
# Usage: bash run_supplementary_folds.sh
set -e

source /usr/local/miniforge3/etc/profile.d/conda.sh

NNUNET_RAW=/home/ge.polymtl.ca/p123239/data/nnUNetv2/nnUNet_raw
NNUNET_PREP=/home/ge.polymtl.ca/p123239/data/nnUNetv2/nnUNet_preprocessed
NNUNET_RES=/home/ge.polymtl.ca/p123239/data/nnUNetv2/nnUNet_results
IMAGES_TS=$NNUNET_RAW/Dataset026_BrnoSpineAll/imagesTs
LABELS_TS=$NNUNET_RAW/Dataset026_BrnoSpineAll/labelsTs
DATASET_JSON=$NNUNET_RAW/Dataset026_BrnoSpineAll/dataset.json
LOG_DIR=/home/ge.polymtl.ca/p123239/SpineFoundation/logs
GPU=0

export nnUNet_raw=$NNUNET_RAW
export nnUNet_preprocessed=$NNUNET_PREP
export nnUNet_results=$NNUNET_RES
export CUDA_VISIBLE_DEVICES=$GPU
export nnUNet_n_proc_DA=8
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export PYTHONUNBUFFERED=1

mkdir -p $LOG_DIR

# ── Étape 1 : générer les folds 9–12 dans splits_final.json ──────────────────
echo "Génération des folds supplémentaires (6, 7, 8, 9 sujets)..."
conda activate dino
cd /home/ge.polymtl.ca/p123239/SpineFoundation
python mri_foundation/make_subject_splits.py --append 6 7 8 9

# ── Étape 2 : entraîner et évaluer chaque nouveau fold ───────────────────────
# fold 9 = 6 sujets, fold 10 = 7 sujets, fold 11 = 8 sujets, fold 12 = 9 sujets
for fold in 9 10 11 12; do
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
echo "Folds supplémentaires terminés."
echo "=========================================="

python3 - <<'EOF'
import json, re
from pathlib import Path
from collections import defaultdict

RES = Path("/home/ge.polymtl.ca/p123239/data/nnUNetv2/nnUNet_results")

def subject_dice(summary_path):
    with open(summary_path) as f:
        d = json.load(f)
    subj = defaultdict(lambda: {'TP':0,'FP':0,'FN':0})
    for case in d['metric_per_case']:
        m = re.search(r'(sub-[^_]+)', case['prediction_file'])
        if not m: continue
        s = m.group(1)
        mets = case['metrics'].get('1', {})
        subj[s]['TP'] += mets.get('TP', 0)
        subj[s]['FP'] += mets.get('FP', 0)
        subj[s]['FN'] += mets.get('FN', 0)
    dices = [2*v['TP']/(2*v['TP']+v['FP']+v['FN']) for v in subj.values() if 2*v['TP']+v['FP']+v['FN'] > 0]
    return sum(dices)/len(dices) if dices else float('nan')

n_subj = {9:6, 10:7, 11:8, 12:9}
print(f"{'Fold':>5} {'N train':>8} {'nnUNet':>8} {'MedDINOv3':>10}")
print('-'*36)
for fold in [9, 10, 11, 12]:
    nnu = subject_dice(RES / f"predictions_nnunet_fold{fold}/summary.json")
    med = subject_dice(RES / f"predictions_meddinov3_fold{fold}/summary.json")
    print(f"{fold:>5} {n_subj[fold]:>8} {nnu:>8.4f} {med:>10.4f}")
EOF
