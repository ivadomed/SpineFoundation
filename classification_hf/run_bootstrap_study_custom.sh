#!/usr/bin/env bash
# Bootstrap study with custom backbone configs.
#
# Usage (from SpineFoundation/):
#   bash classification_hf/run_bootstrap_study_custom.sh [n_runs] [n_bootstrap] [n_parallel]
#
# Defaults: 5 runs/task, 1000 bootstrap resamples, 6 parallel jobs
# Each parallel job runs on a single GPU (round-robin across available GPUs).
# Bootstrap analysis for all 3 tasks runs in parallel.

set -eu

ROOT=/home/ge.polymtl.ca/p123239/SpineFoundation
N_RUNS="${1:-5}"
N_BOOTSTRAP="${2:-1000}"
N_PARALLEL="${3:-6}"   # jobs in parallel — safe to go up to N_RUNS*3 (48 cores, 125GB RAM)

PYTHON=/home/ge.polymtl.ca/p123239/.conda/envs/dino/bin/python
CONFIGS=(
    "$ROOT/classification_hf/configs/rsna_nfn_custom.yaml"
    "$ROOT/classification_hf/configs/rsna_scs_custom.yaml"
    "$ROOT/classification_hf/configs/rsna_ss_custom.yaml"
)
OUTPUT_DIRS=(
    "$ROOT/outputs_cls/rsna_nfn_custom"
    "$ROOT/outputs_cls/rsna_scs_custom"
    "$ROOT/outputs_cls/rsna_ss_custom"
)
TASKS=("nfn" "scs" "ss")
LOG_DIR="$ROOT/classification_hf/logs_custom"

cd "$ROOT"

echo "================================================================"
echo "Bootstrap study (custom backbone) — $(date)"
echo "  Runs/task  : $N_RUNS"
echo "  Bootstraps : $N_BOOTSTRAP"
echo "  Parallel   : $N_PARALLEL jobs at a time (round-robin GPU 0/1)"
echo "================================================================"

# ── Phase 1 : training runs — N_PARALLEL workers (round-robin GPUs) ──────────
# Each run uses a single GPU (features are pre-cached → no backbone needed).

TOTAL_RUNS=$(( ${#CONFIGS[@]} * N_RUNS ))
echo "Total runs: $TOTAL_RUNS"

# Build the full list of (config, task, seed, gpu) jobs
CONFIGS_LIST=()
TASKS_LIST=()
SEEDS_LIST=()
GPUS_LIST=()

job_idx=0
for i in "${!CONFIGS[@]}"; do
    for run in $(seq 1 $N_RUNS); do
        SEED=$(( run * 13 + 37 ))
        GPU=$(( job_idx % 2 ))
        CONFIGS_LIST+=("${CONFIGS[$i]}")
        TASKS_LIST+=("${TASKS[$i]}")
        SEEDS_LIST+=("$SEED")
        GPUS_LIST+=("$GPU")
        job_idx=$(( job_idx + 1 ))
    done
done

# Run N_PARALLEL at a time
overall=0
while [ $overall -lt $TOTAL_RUNS ]; do
    pids=()
    for slot in $(seq 0 $(( N_PARALLEL - 1 ))); do
        idx=$(( overall + slot ))
        [ $idx -ge $TOTAL_RUNS ] && break

        CONFIG="${CONFIGS_LIST[$idx]}"
        TASK="${TASKS_LIST[$idx]}"
        SEED="${SEEDS_LIST[$idx]}"
        GPU="${GPUS_LIST[$idx]}"

        echo ""
        echo "[$(date +%H:%M:%S)]  Run $((idx+1))/$TOTAL_RUNS  task=$TASK  seed=$SEED  gpu=$GPU"

        CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" \
            -m classification_hf.train \
            --config "$CONFIG" \
            --set seed=$SEED &
        pids+=($!)
    done

    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    overall=$(( overall + N_PARALLEL ))
done

echo ""
echo "================================================================"
echo "All training runs complete.  Starting bootstrap analysis …"
echo "================================================================"

# Clear previous results so each run of the script produces a fresh CSV
rm -f "$LOG_DIR/bootstrap_results.csv" "$LOG_DIR/bootstrap_pretrained_results.csv"

# ── Phase 2 : bootstrap analysis — all 3 tasks in parallel ───────────────────
bootstrap_pids=()
for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    RUNS_DIR="${OUTPUT_DIRS[$i]}"

    echo ""
    echo "[$(date +%H:%M:%S)]  Bootstrap: task=$TASK  runs_dir=$RUNS_DIR"

    "$PYTHON" -m classification_hf.bootstrap_eval \
        --task         "$TASK" \
        --runs-dir     "$RUNS_DIR" \
        --n-bootstrap  "$N_BOOTSTRAP" \
        --log-dir      "$LOG_DIR" &
    bootstrap_pids+=($!)
done

for pid in "${bootstrap_pids[@]}"; do
    wait "$pid"
done

echo ""
echo "================================================================"
echo "Bootstrap study complete — $(date)"
echo "Results:"
echo "  CSV    : $LOG_DIR/bootstrap_results.csv"
echo "  JSON   : $LOG_DIR/bootstrap_{nfn,scs,ss}.json"
echo "================================================================"
