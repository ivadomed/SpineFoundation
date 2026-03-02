#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Two-step classification pipeline
#   Step 1: extract frozen backbone features (run once)
#   Step 2: train lightweight classification head on cached features
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PYTHON=/home/ge.polymtl.ca/p123239/.conda/envs/dino/bin/python
ROOT=/home/ge.polymtl.ca/p123239/SpineFoundation

MODEL_DIR=/home/ge.polymtl.ca/p123239/SpineFoundation/curia_model/models--raidium--curia/snapshots/9657dc56276bc6c9503ef6f8d060879c8bee482f
DATA_DIR=/home/ge.polymtl.ca/p123239/data/RSNA_patches_512
FEATURES_DIR=/home/ge.polymtl.ca/p123239/data/RSNA_patches_512_features
OUTPUT_DIR=/home/ge.polymtl.ca/p123239/SpineFoundation/outputs_cls/rsna_curia

cd "$ROOT"

# ── Step 1: Feature extraction ────────────────────────────────────────────────
# Skip this step if features already exist.
if [ ! -f "$FEATURES_DIR/train.npz" ] || [ ! -f "$FEATURES_DIR/val.npz" ]; then
    echo "=== Step 1: Extracting features ==="
    $PYTHON -m classification_hf.extract_features \
        --model_dir  "$MODEL_DIR" \
        --data_dir   "$DATA_DIR" \
        --output_dir "$FEATURES_DIR" \
        --image_size 512 \
        --batch_size 64 \
        --num_workers 8 \
        --val_split 0.2 \
        --seed 42 \
        --amp
else
    echo "=== Step 1: Features already extracted, skipping ==="
fi

# ── Step 2: Train classifier ──────────────────────────────────────────────────
echo "=== Step 2: Training classification head ==="
$PYTHON -m classification_hf.train_cls \
    --features_dir "$FEATURES_DIR" \
    --output_dir   "$OUTPUT_DIR" \
    --epochs 200 \
    --batch_size 512 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --num_workers 4 \
    --seed 42 \
    --use_class_weights \
    --hidden_dim 256 \
    --dropout 0.2 \
    --wandb \
    --wandb_project spine-cls \
    --wandb_mode online
