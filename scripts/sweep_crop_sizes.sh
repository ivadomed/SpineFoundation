#!/usr/bin/env bash
# Sweep 8 crop sizes (0=full slice → 50×50) through RSNAextractor + eval_pretrained_nfn
# Results are saved in logs/crop_sweep/crop_<size>.txt

set -euo pipefail

ROOT="$HOME/data_ok/lumbar-rsna-challenge-2024"
DATA_BASE="$HOME/data/RSNA_patches_scs_sweep"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs/crop_sweep"
PYTHON="${PYTHON:-python}"

CROP_SIZES=(0 500 400 300 200 150 100 50)

mkdir -p "$LOG_DIR"

echo "=============================="
echo " Crop sweep — $(date)"
echo " Root      : $ROOT"
echo " Data base : $DATA_BASE"
echo " Log dir   : $LOG_DIR"
echo "=============================="

for SIZE in "${CROP_SIZES[@]}"; do
    if [ "$SIZE" -eq 0 ]; then
        LABEL="full"
    else
        LABEL="crop${SIZE}"
    fi

    SLICE_DIR="${DATA_BASE}/${LABEL}"
    LOG_FILE="${LOG_DIR}/${LABEL}.txt"

    echo ""
    echo "----------------------------------------------"
    echo " Crop size : ${SIZE}  →  ${SLICE_DIR}"
    echo "----------------------------------------------"

    # ── 1. Extract slices (skip if already done) ──────────────────────────────
    if [ -d "$SLICE_DIR" ] && [ "$(find "$SLICE_DIR" -name '*.npz' | wc -l)" -gt 0 ]; then
        echo "[skip] Slice dir already exists with NPZ files."
    else
        echo "[extract] Running RSNAextractor.py ..."
        $PYTHON "$SCRIPT_DIR/RSNAextractor.py" \
            --root   "$ROOT" \
            --out-dir "$SLICE_DIR" \
            --crop-size "$SIZE"
    fi

    # ── 2. Evaluate ────────────────────────────────────────────────────────────
    echo "[eval] Running eval_pretrained_nfn.py  →  $LOG_FILE"
    $PYTHON "$SCRIPT_DIR/eval_pretrained_nfn.py" \
        --data-dir "$SLICE_DIR" \
        2>&1 | tee "$LOG_FILE"

    echo "[done] Logged to $LOG_FILE"
done

# ── Summary table ─────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  RÉSUMÉ AUC OvR macro par crop size"
echo "============================================================"
printf "  %-12s  %s\n" "crop_size" "AUC_macro"
echo "  ------------  ----------"
for SIZE in "${CROP_SIZES[@]}"; do
    if [ "$SIZE" -eq 0 ]; then
        LABEL="full"
    else
        LABEL="crop${SIZE}"
    fi
    LOG_FILE="${LOG_DIR}/${LABEL}.txt"
    AUC=$(grep -oP "macro\s*:\s*\K[0-9.]+" "$LOG_FILE" | head -1 || echo "N/A")
    printf "  %-12s  %s\n" "$LABEL" "$AUC"
done
echo "============================================================"
echo "Logs complets dans : $LOG_DIR"
