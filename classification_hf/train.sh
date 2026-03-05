#!/usr/bin/env bash
# Usage:
#   bash classification_hf/train.sh [config] [n_gpus]
#
# Examples (run from SpineFoundation/):
#   bash classification_hf/train.sh classification_hf/configs/rsna_neural_foraminal_narrowing.yaml
#   bash classification_hf/train.sh classification_hf/configs/rsna_spinal_canal_stenosis.yaml 2
set -eu

ROOT=/home/ge.polymtl.ca/p123239/SpineFoundation
CONFIG="${1:-$ROOT/classification_hf/configs/rsna_neural_foraminal_narrowing.yaml}"
N_GPUS="${2:-1}"

cd "$ROOT"

echo "[$(date +%H:%M:%S)]  Config  : $CONFIG"
echo "[$(date +%H:%M:%S)]  N GPUs  : $N_GPUS"

PYTHON=/home/ge.polymtl.ca/p123239/.conda/envs/dino/bin/python

"$PYTHON" -m torch.distributed.run \
    --nproc_per_node="$N_GPUS" \
    --standalone \
    -m classification_hf.train \
    --config "$CONFIG"
