#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BIDS_ROOT="/home/ge.polymtl.ca/p123239/data_ok/lumbar-rsna-challenge-2024"
DATA_ROOT="/home/ge.polymtl.ca/p123239/data"
MODEL_NAME="raidium/curia"
CROP_SIZE=0        # 0 = coupe entière (raw), sinon ex. 200
BATCH_SIZE=64
LOG_DIR="logs"

TASKS=("nfn" "ss" "scs")

# ── Étape 1 — Extraction des patches ──────────────────────────────────────────
echo "========================================================"
echo " ÉTAPE 1 — Extraction des patches (RSNAextractor.py)"
echo "========================================================"

for TASK in "${TASKS[@]}"; do
    OUT_DIR="${DATA_ROOT}/patches_RSNA_raw_with_mask_${TASK}"
    echo ""
    echo ">>> Extraction : --task ${TASK}  -->  ${OUT_DIR}"
    python "${SCRIPT_DIR}/RSNAextractor.py" \
        --task      "${TASK}"      \
        --root      "${BIDS_ROOT}" \
        --out-dir   "${OUT_DIR}"   \
        --crop-size "${CROP_SIZE}"
done

# ── Étape 2 — Évaluation ──────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " ÉTAPE 2 — Évaluation (eval_pretrained.py)"
echo "========================================================"

for TASK in "${TASKS[@]}"; do
    echo ""
    echo ">>> Évaluation : --task ${TASK}"
    python "${SCRIPT_DIR}/eval_pretrained.py" \
        --task        "${TASK}"       \
        --model-name  "${MODEL_NAME}" \
        --batch-size  "${BATCH_SIZE}" \
        --log-dir     "${LOG_DIR}"
done

echo ""
echo "========================================================"
echo " Pipeline terminé."
echo "========================================================"
