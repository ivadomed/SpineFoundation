#!/bin/bash
PYTHON=/home/ge.polymtl.ca/p123239/.conda/envs/FM/bin/python
DINOV3=/home/ge.polymtl.ca/p123239/FM/models/dinov3-vitl16
DATA=/home/ge.polymtl.ca/p123239/SpineFoundation/segmentation_hf/data/dinov3
OUT=/home/ge.polymtl.ca/p123239/SpineFoundation/outputs_seg_dinov3

for fold in 0 1 2 3 4 5 6 7 8 9 10 11; do
    DONE_MARKER="$OUT/tune_fold${fold}/final/best.pt"
    if [ -f "$DONE_MARKER" ]; then
        echo "[fold ${fold}] Déjà terminé, skip."
        continue
    fi

    LOG_DIR="$(dirname $0)/segmentation_hf/logs_seg_folds"
    mkdir -p "$LOG_DIR"
    echo "[fold ${fold}] Lancement..."
    for attempt in 1 2 3; do
        CUDA_VISIBLE_DEVICES=1 $PYTHON -m segmentation_hf.tune \
            --model_dir $DINOV3 \
            --npz_train_dir $DATA/fold_${fold}/train_npz \
            --npz_val_dir   $DATA/fold_${fold}/val_npz \
            --output_dir    $OUT/tune_fold${fold} \
            --n_trials 30 --trial_epochs 30 --final_epochs 300 --amp --batch_size 256 \
            2>&1 | tee -a "$LOG_DIR/dinov3_fold_${fold}.log" \
            && break \
            || echo "[fold ${fold}] CRASH (tentative $attempt/3), retry dans 10s..."
        sleep 10
    done
    [ -f "$OUT/tune_fold${fold}/final/best.pt" ] \
        && echo "[fold ${fold}] Terminé." \
        || echo "[fold ${fold}] ÉCHEC après 3 tentatives."
    sleep 5
done

echo "Tous les folds DINOv3 terminés."
