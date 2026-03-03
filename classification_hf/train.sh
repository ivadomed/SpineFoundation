#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/ge.polymtl.ca/p123239/.conda/envs/dino/bin/python
ROOT=/home/ge.polymtl.ca/p123239/SpineFoundation
CONFIG=$ROOT/classification_hf/configs/rsna_neural_foraminal_narrowing.yaml

cd "$ROOT"

$PYTHON -m classification_hf.train --config "$CONFIG"
