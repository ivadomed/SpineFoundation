#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=1

PYTHON=/home/ge.polymtl.ca/p123239/.conda/envs/FM/bin/python
REPO=/home/ge.polymtl.ca/p123239/SpineFoundation
DATA=/home/ge.polymtl.ca/p123239/data/RSNA_patches_scs
MODEL=/home/ge.polymtl.ca/p123239/.cache/huggingface/hub/models--raidium--curia/snapshots/9657dc56276bc6c9503ef6f8d060879c8bee482f

N_TRIALS=50
TRIAL_EPOCHS=50
FINAL_EPOCHS=300

cd "$REPO"

# 1. Cache patch tokens + CLS tokens (single GPU pass, skip if already cached)
echo "==> Cache patch tokens (crop 4cm)"
"$PYTHON" -m classification_hf.cache_patch_tokens \
    --data_dir "$DATA" \
    --model_name "$MODEL" \
    --suffix crop4cm \
    --crop_cm 4.0

echo "==> Cache CLS features"
"$PYTHON" -m classification_hf.cache_pooled_features \
    --data_dir "$DATA" \
    --token_key cls_token_curia_crop4cm \
    --cache_suffix curia_crop4cm_cls

# 2. Un fold par régime (1 split par taille de training set)
# regime_50_split_1_set ... regime_all_split_1_set → 9 folds, courbe d'apprentissage
FOLDS="regime_50_split_1_set \
       regime_100_split_1_set \
       regime_200_split_1_set \
       regime_300_split_1_set \
       regime_400_split_1_set \
       regime_500_split_1_set \
       regime_750_split_1_set \
       regime_1000_split_1_set \
       regime_all_split_1_set"

for FOLD in $FOLDS; do
    echo ""
    echo "========================================"
    echo "==> Fold: $FOLD"
    echo "========================================"

    # Resnet (TokenGridClassifier)
    echo "  -> Tune resnet"
    "$PYTHON" -m classification_hf.tune \
        --config classification_hf/configs/rsna_scs_crop4cm_resnet.yaml \
        --output_dir outputs_cls/tune_scs_resnet/"$FOLD" \
        --n_trials $N_TRIALS \
        --trial_epochs $TRIAL_EPOCHS \
        --final_epochs $FINAL_EPOCHS \
        --set fold_column="$FOLD"

    # CLS linear
    echo "  -> Tune CLS linear"
    "$PYTHON" -m classification_hf.tune \
        --config classification_hf/configs/rsna_scs_crop4cm_cls.yaml \
        --output_dir outputs_cls/tune_scs_cls/"$FOLD" \
        --n_trials $N_TRIALS \
        --trial_epochs $TRIAL_EPOCHS \
        --final_epochs $FINAL_EPOCHS \
        --set fold_column="$FOLD"
done

echo ""
echo "==> Done — 9 régimes (learning curve) completed"
