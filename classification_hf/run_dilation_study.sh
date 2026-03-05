#!/usr/bin/env bash
# Dilation-radius ablation study — Curia backbone, all 3 tasks
#
# Usage (from SpineFoundation/):
#   bash classification_hf/run_dilation_study.sh [n_gpus]
#
# For each dilation radius × task combination, one training run is launched.
# Results are appended to classification_hf/logs/results.csv automatically.
# After all runs, the plot script is called to produce the summary figure.
#
# Time estimate (2 GPUs, 35 epochs each):
#   8 radii × (nfn 2min + scs 1min + ss 2min) ≈ 40 min total

set -eu

ROOT=/home/ge.polymtl.ca/p123239/SpineFoundation
N_GPUS="${1:-2}"
PYTHON=/home/ge.polymtl.ca/p123239/.conda/envs/dino/bin/python

# Dilation radii to sweep
RADII=(0 2 4 6 8 12 16 24)

TASKS=(   "nfn"   "scs"   "ss"  )
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

TOTAL=$(( ${#RADII[@]} * ${#TASKS[@]} ))
RUN_IDX=0

cd "$ROOT"

echo "================================================================"
echo "Dilation study — $(date)"
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
        RUN_IDX=$(( RUN_IDX + 1 ))

        echo ""
        echo "[$(date +%H:%M:%S)]  Run $RUN_IDX/$TOTAL  task=$TASK  dilation=$DIL"

        "$PYTHON" -m torch.distributed.run \
            --nproc_per_node="$N_GPUS" \
            --standalone \
            -m classification_hf.train \
            --config "$CONFIG" \
            --set \
                dilation_radius=$DIL \
                task="${TASK}_dil${DIL}" \
                output_dir="${OUT_BASE}_dil${DIL}"
    done
done

echo ""
echo "================================================================"
echo "All runs done. Generating plots …"
echo "================================================================"

"$PYTHON" -m classification_hf.plot_dilation_study \
    --log_dir "$ROOT/classification_hf/logs" \
    --tasks "${TASKS[@]}" \
    --radii "${RADII[@]}"

echo ""
echo "================================================================"
echo "Dilation study complete — $(date)"
echo "  Results : $ROOT/classification_hf/logs/results.csv"
echo "  Plot    : $ROOT/classification_hf/logs/dilation_study.png"
echo "================================================================"
