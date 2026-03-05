#!/usr/bin/env bash
# Bootstrap study: 5 training runs per task → bootstrap 95% CI evaluation
#
# Usage (from SpineFoundation/):
#   bash classification_hf/run_bootstrap_study.sh [n_gpus] [n_runs] [n_bootstrap]
#
# Defaults: 2 GPUs, 5 runs, 1000 bootstrap resamples
#
# Time estimate (with 2 GPUs, 35 epochs):
#   5 × (nfn 2min + scs 1min + ss 2min) ≈ 25 min training
#   Bootstrap: ~5 seconds per task
#   Total: ~30 minutes

set -eu

ROOT=/home/ge.polymtl.ca/p123239/SpineFoundation
N_GPUS="${1:-2}"
N_RUNS="${2:-5}"
N_BOOTSTRAP="${3:-1000}"

PYTHON=/home/ge.polymtl.ca/p123239/.conda/envs/dino/bin/python
CONFIGS=(
    "$ROOT/classification_hf/configs/rsna_neural_foraminal_narrowing.yaml"
    "$ROOT/classification_hf/configs/rsna_spinal_canal_stenosis.yaml"
    "$ROOT/classification_hf/configs/rsna_subarticular_stenosis.yaml"
)
OUTPUT_DIRS=(
    "$ROOT/outputs_cls/rsna_nfn"
    "$ROOT/outputs_cls/rsna_scs"
    "$ROOT/outputs_cls/rsna_ss"
)
TASKS=("nfn" "scs" "ss")
LOG_DIR="$ROOT/classification_hf/logs"

cd "$ROOT"

echo "================================================================"
echo "Bootstrap study — $(date)"
echo "  GPUs       : $N_GPUS"
echo "  Runs/task  : $N_RUNS"
echo "  Bootstraps : $N_BOOTSTRAP"
echo "================================================================"

# ── Phase 1 : training runs ───────────────────────────────────────────────────
TOTAL_RUNS=$(( ${#CONFIGS[@]} * N_RUNS ))
RUN_IDX=0

for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    TASK="${TASKS[$i]}"

    echo ""
    echo "── Task: $TASK  (config: $(basename $CONFIG)) ──"

    for run in $(seq 1 $N_RUNS); do
        RUN_IDX=$(( RUN_IDX + 1 ))
        SEED=$(( run * 13 + 37 ))   # deterministic but distinct: 50, 63, 76, 89, 102
        echo ""
        echo "[$(date +%H:%M:%S)]  Run $run/$N_RUNS for task=$TASK  seed=$SEED  (overall $RUN_IDX/$TOTAL_RUNS)"

        "$PYTHON" -m torch.distributed.run \
            --nproc_per_node="$N_GPUS" \
            --standalone \
            -m classification_hf.train \
            --config "$CONFIG" \
            --set seed=$SEED
    done
done

echo ""
echo "================================================================"
echo "All training runs complete.  Starting bootstrap analysis …"
echo "================================================================"

# ── Phase 2 : bootstrap analysis ─────────────────────────────────────────────
for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    RUNS_DIR="${OUTPUT_DIRS[$i]}"

    echo ""
    echo "[$(date +%H:%M:%S)]  Bootstrap: task=$TASK  runs_dir=$RUNS_DIR"

    "$PYTHON" -m classification_hf.bootstrap_eval \
        --task       "$TASK" \
        --runs_dir   "$RUNS_DIR" \
        --n_bootstrap "$N_BOOTSTRAP" \
        --log_dir    "$LOG_DIR"
done

echo ""
echo "================================================================"
echo "Bootstrap study complete — $(date)"
echo "Results:"
echo "  CSV    : $LOG_DIR/bootstrap_results.csv"
echo "  JSON   : $LOG_DIR/bootstrap_{nfn,scs,ss}.json"
echo "================================================================"
