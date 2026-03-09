#!/usr/bin/env bash
# Dilation-radius ablation study — Curia backbone, all 3 tasks
#
# Usage (from SpineFoundation/):
#   bash classification_hf/run_dilation_study.sh [--custom] [n_gpus]
#
#   --custom   Use custom backbone configs (cache_suffix=custom,
#              loads pooled_features_custom_dil{N}.pt)
#
# For each dilation radius × task combination:
#   1. Train a model from scratch with torch.distributed.run
#   2. Run bootstrap_eval.py --mode trained to compute 95% CIs on val predictions
# Results are appended to classification_hf/logs/bootstrap_results.csv.
# After all runs, the plot script generates the summary figure.

set -eu

ROOT=/home/ge.polymtl.ca/p123239/SpineFoundation
PYTHON=/home/ge.polymtl.ca/p123239/.conda/envs/dino/bin/python

# ── Parse args ────────────────────────────────────────────────────────────────
CUSTOM=false
N_GPUS=2
for arg in "$@"; do
    case "$arg" in
        --custom) CUSTOM=true ;;
        [0-9]*)   N_GPUS="$arg" ;;
    esac
done

# Dilation radii to sweep
RADII=(0 8 16 32 48 64)

TASKS=( "nfn"  "scs" "ss" )

if $CUSTOM; then
    CONFIGS=(
        "$ROOT/classification_hf/configs/rsna_nfn_custom.yaml"
        "$ROOT/classification_hf/configs/rsna_scs_custom.yaml"
        "$ROOT/classification_hf/configs/rsna_ss_custom.yaml"
    )
    OUTPUT_BASES=(
        "$ROOT/outputs_cls/rsna_nfn_custom"
        "$ROOT/outputs_cls/rsna_scs_custom"
        "$ROOT/outputs_cls/rsna_ss_custom"
    )
    VARIANT="custom"
    LOG_DIR="$ROOT/classification_hf/logs_custom"
else
    CONFIGS=(
        "$ROOT/classification_hf/configs/rsna_neural_foraminal_narrowing.yaml"
        "$ROOT/classification_hf/configs/rsna_spinal_canal_stenosis.yaml"
        "$ROOT/classification_hf/configs/rsna_subarticular_stenosis.yaml"
    )
    OUTPUT_BASES=(
        "$ROOT/outputs_cls/rsna_nfn"
        "$ROOT/outputs_cls/rsna_scs"
        "$ROOT/outputs_cls/rsna_ss"
    )
    VARIANT="default"
    LOG_DIR="$ROOT/classification_hf/logs"
fi

TOTAL=$(( ${#RADII[@]} * ${#TASKS[@]} ))
RUN_IDX=0

cd "$ROOT"

echo "================================================================"
echo "Dilation study — $(date)"
echo "  Variant: $VARIANT"
echo "  Radii  : ${RADII[*]}"
echo "  Tasks  : ${TASKS[*]}"
echo "  GPUs   : $N_GPUS"
echo "  Total  : $TOTAL runs"
echo "================================================================"

for DIL in "${RADII[@]}"; do
    for i in "${!TASKS[@]}"; do
        TASK="${TASKS[$i]}"
        CONFIG="${CONFIGS[$i]}"
        OUT_BASE="${OUTPUT_BASES[$i]}"
        OUT_DIR="${OUT_BASE}_dil${DIL}"
        RUN_IDX=$(( RUN_IDX + 1 ))

        echo ""
        echo "[$(date +%H:%M:%S)]  Run $RUN_IDX/$TOTAL  task=$TASK  dilation=$DIL  variant=$VARIANT"

        # Step 1 — train from scratch
        "$PYTHON" -m torch.distributed.run \
            --nproc_per_node="$N_GPUS" \
            --standalone \
            -m classification_hf.train \
            --config "$CONFIG" \
            --set \
                dilation_radius=$DIL \
                task="${TASK}_dil${DIL}" \
                output_dir="$OUT_DIR"

        # Step 2 — bootstrap evaluation on the saved val_predictions.npz
        "$PYTHON" -m classification_hf.bootstrap_eval \
            --mode trained \
            --task "$TASK" \
            --dilation-radius "$DIL" \
            --runs-dir "$OUT_DIR" \
            --log-dir "$LOG_DIR"
    done
done

echo ""
echo "================================================================"
echo "All runs done. Generating plots …"
echo "================================================================"

# Plot this variant alone
"$PYTHON" -m classification_hf.plot_dilation_study \
    --log_dir "$LOG_DIR" \
    --tasks "${TASKS[@]}" \
    --radii "${RADII[@]}"

# Comparison plot (only if both log dirs exist)
DEFAULT_LOG="$ROOT/classification_hf/logs"
CUSTOM_LOG="$ROOT/classification_hf/logs_custom"
if [ -f "$DEFAULT_LOG/bootstrap_results.csv" ] && [ -f "$CUSTOM_LOG/bootstrap_results.csv" ]; then
    "$PYTHON" -m classification_hf.plot_dilation_study \
        --log_dir "$DEFAULT_LOG" \
        --compare_log_dir "$CUSTOM_LOG" \
        --labels "default" "custom" \
        --tasks "${TASKS[@]}" \
        --radii "${RADII[@]}" \
        --out "$DEFAULT_LOG/dilation_study_comparison.png"
fi

echo ""
echo "================================================================"
echo "Dilation study complete — $(date)  [$VARIANT]"
echo "  Results : $LOG_DIR/bootstrap_results.csv"
echo "  Plot    : $LOG_DIR/dilation_study.png"
echo "================================================================"
