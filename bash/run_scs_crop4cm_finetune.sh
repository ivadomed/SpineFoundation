#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=1

PYTHON=/home/ge.polymtl.ca/p123239/.conda/envs/FM/bin/python
REPO=/home/ge.polymtl.ca/p123239/SpineFoundation

N_TRIALS=50
TRIAL_EPOCHS=20
FINAL_EPOCHS=100

cd "$REPO"

FOLDS="regime_50_split_1_set \
       regime_100_split_1_set \
       regime_200_split_1_set \
       regime_300_split_1_set \
       regime_400_split_1_set \
       regime_500_split_1_set \
       regime_750_split_1_set \
       regime_all_split_1_set"

for FOLD in $FOLDS; do
    echo ""
    echo "========================================"
    echo "==> Fold: $FOLD"
    echo "========================================"

    "$PYTHON" -m classification_hf.tune \
        --config classification_hf/configs/rsna_scs_crop4cm_finetune.yaml \
        --output_dir outputs_cls/tune_scs_finetune/"$FOLD" \
        --metric wbce \
        --n_trials $N_TRIALS \
        --trial_epochs $TRIAL_EPOCHS \
        --final_epochs $FINAL_EPOCHS \
        --set fold_column="$FOLD"
done

echo ""
echo "==> Done — 8 régimes (unfrozen backbone) completed"
