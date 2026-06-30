#!/usr/bin/env bash
# Usage:
#   bash classification_hf/train.sh <config> [--set key=val ...]
#
# Examples (run from SpineFoundation/):
#   bash classification_hf/train.sh classification_hf/configs/rsna_nfn_fold.yaml
#   bash classification_hf/train.sh classification_hf/configs/rsna_nfn_fold.yaml --set fold_column=regime_100_split_3_set
set -eu

ROOT=/home/ge.polymtl.ca/p123239/SpineFoundation
CONFIG="${1:-$ROOT/classification_hf/configs/rsna_neural_foraminal_narrowing.yaml}"
# All remaining args forwarded to Python (e.g. --set fold_column=regime_50_split_1_set)
shift
EXTRA_ARGS="$@"

cd "$ROOT"

echo "[$(date +%H:%M:%S)]  Config  : $CONFIG"

python -m classification_hf.train --config "$CONFIG" $EXTRA_ARGS
