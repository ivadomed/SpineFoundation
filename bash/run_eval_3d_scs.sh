#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=1

PYTHON=/home/ge.polymtl.ca/p123239/.conda/envs/FM/bin/python
REPO=/home/ge.polymtl.ca/p123239/SpineFoundation

cd "$REPO"

"$PYTHON" -m classification_hf.claude_scs \
    --data      /home/ge.polymtl.ca/p123239/data/RSNA_patches_3D \
    --model_path classification_hf \
    --fold_csv  /home/ge.polymtl.ca/p123239/fold_split_RSNA.json \
    --gt_dir    /home/ge.polymtl.ca/p123239/data/RSNA_patches_scs \
    --output_csv outputs_cls/claude_scs_test_predictions.csv

echo "==> Done"
