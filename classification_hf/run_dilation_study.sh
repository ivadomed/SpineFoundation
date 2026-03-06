#!/usr/bin/env bash
# Dilation-radius ablation study — Curia backbone, all 3 tasks (bootstrap mode)
#
# Usage (from SpineFoundation/):
#   bash classification_hf/run_dilation_study.sh
#
# For each dilation radius × task combination, bootstrap evaluation is run
# using the pretrained Curia head on cached patch_tokens.
# Results are appended to classification_hf/logs/bootstrap_pretrained_results.csv.

set -eu

ROOT=/home/ge.polymtl.ca/p123239/SpineFoundation
PYTHON=/home/ge.polymtl.ca/p123239/.conda/envs/dino/bin/python

# Dilation radii to sweep
RADII=(0 2 4 6 8 12 16 24)

TASKS=( "nfn" "scs" "ss" )

TOTAL=$(( ${#RADII[@]} * ${#TASKS[@]} ))
RUN_IDX=0

cd "$ROOT"

echo "================================================================"
echo "Dilation bootstrap study — $(date)"
echo "  Radii  : ${RADII[*]}"
echo "  Tasks  : ${TASKS[*]}"
echo "  Total  : $TOTAL runs"
echo "================================================================"

for DIL in "${RADII[@]}"; do
    for TASK in "${TASKS[@]}"; do
        RUN_IDX=$(( RUN_IDX + 1 ))

        echo ""
        echo "[$(date +%H:%M:%S)]  Run $RUN_IDX/$TOTAL  task=$TASK  dilation=$DIL"

        "$PYTHON" -m classification_hf.bootstrap_eval \
            --mode pretrained \
            --task "$TASK" \
            --dilation-radius "$DIL" \
            --log-dir "$ROOT/classification_hf/logs"
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
echo "Dilation bootstrap study complete — $(date)"
echo "  Results : $ROOT/classification_hf/logs/bootstrap_pretrained_results.csv"
echo "  Plot    : $ROOT/classification_hf/logs/dilation_study.png"
echo "================================================================"
