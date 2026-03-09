#!/usr/bin/env bash
# Bootstrap study for teacher checkpoint iter_100000.
#
# Usage (from SpineFoundation/):
#   bash classification_hf/run_bootstrap_iter100k.sh [n_runs] [n_bootstrap] [n_parallel]
#
# Defaults: 5 runs/task, 1000 bootstrap resamples, 6 parallel jobs
#
# Steps:
#   1. Cache patch tokens (GPU) using iter_100000 backbone → patch_tokens_iter100k in NPZ
#   2. Pool features (CPU)                                 → pooled_features_iter100k_dil0.pt
#   3. Generate YAML configs
#   4. Training runs (linear head only, no backbone needed)
#   5. Bootstrap analysis

set -eu

ROOT=/home/ge.polymtl.ca/p123239/SpineFoundation
N_RUNS="${1:-5}"
N_BOOTSTRAP="${2:-1000}"
N_PARALLEL="${3:-6}"

PYTHON=/home/ge.polymtl.ca/p123239/.conda/envs/dino/bin/python
BACKBONE=/home/ge.polymtl.ca/p123239/SpineFoundation/outputs_curia/custom_curia_512/teacher_checkpoints/iter_100000
PROCESSOR=/home/ge.polymtl.ca/p123239/.cache/huggingface/hub/models--raidium--curia/snapshots/9657dc56276bc6c9503ef6f8d060879c8bee482f
SUFFIX=iter100k
DILATION=0

DATA_DIRS=(
    "/home/ge.polymtl.ca/p123239/data/RSNA_patches_nfn"
    "/home/ge.polymtl.ca/p123239/data/RSNA_patches_scs"
    "/home/ge.polymtl.ca/p123239/data/RSNA_patches_ss"
)
TASKS=("nfn" "scs" "ss")
SUBFOLDERS=("neural_foraminal_narrowing" "spinal_canal_stenosis" "subarticular_stenosis")

CONFIG_DIR="$ROOT/classification_hf/configs"
OUTPUT_BASE="$ROOT/outputs_cls"
LOG_DIR="$ROOT/classification_hf/logs_${SUFFIX}"
LOG_PATCH="$ROOT/classification_hf/logs_${SUFFIX}_patch"

mkdir -p "$LOG_DIR" "$LOG_PATCH"

echo "================================================================"
echo "Bootstrap study (iter_100000) — $(date)"
echo "  Backbone   : $BACKBONE"
echo "  Suffix     : $SUFFIX"
echo "  Dilation   : $DILATION"
echo "  Runs/task  : $N_RUNS"
echo "  Bootstraps : $N_BOOTSTRAP"
echo "  Parallel   : $N_PARALLEL"
echo "================================================================"

cd "$ROOT"

# ── Phase 1 : Cache patch tokens (GPU) ───────────────────────────────────────
echo ""
echo "================================================================"
echo "Phase 1: Caching patch tokens (patch_tokens_${SUFFIX}) …"
echo "================================================================"

for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    DATA_DIR="${DATA_DIRS[$i]}"
    echo ""
    echo "[$(date +%H:%M:%S)]  cache_patch_tokens  task=$TASK  data=$DATA_DIR"
    "$PYTHON" -m classification_hf.cache_patch_tokens \
        --data_dir       "$DATA_DIR" \
        --model_name     "$BACKBONE" \
        --processor_name "$PROCESSOR" \
        --suffix         "$SUFFIX" \
        --batch_size     64 \
        2>&1 | tee "$LOG_PATCH/patch_tokens_${TASK}.log"
done

echo ""
echo "Phase 1 complete."

# ── Phase 2 : Pool features (CPU) ────────────────────────────────────────────
echo ""
echo "================================================================"
echo "Phase 2: Pooling features (pooled_features_${SUFFIX}_dil${DILATION}.pt) …"
echo "================================================================"

pool_pids=()
for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    DATA_DIR="${DATA_DIRS[$i]}"
    echo ""
    echo "[$(date +%H:%M:%S)]  cache_pooled_features  task=$TASK"
    "$PYTHON" -m classification_hf.cache_pooled_features \
        --data_dir        "$DATA_DIR" \
        --token_key       "patch_tokens_${SUFFIX}" \
        --cache_suffix    "$SUFFIX" \
        --dilation_radius "$DILATION" \
        --num_workers     16 \
        2>&1 | tee "$LOG_PATCH/pooled_features_${TASK}.log" &
    pool_pids+=($!)
done

for pid in "${pool_pids[@]}"; do
    wait "$pid"
done

echo ""
echo "Phase 2 complete."

# ── Phase 3 : Generate YAML configs ──────────────────────────────────────────
echo ""
echo "================================================================"
echo "Phase 3: Generating configs …"
echo "================================================================"

CONFIGS=()
OUTPUT_DIRS=()

for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    SUBFOLDER="${SUBFOLDERS[$i]}"
    DATA_DIR="${DATA_DIRS[$i]}"
    CONFIG_PATH="$CONFIG_DIR/rsna_${TASK}_${SUFFIX}.yaml"
    OUTPUT_DIR="$OUTPUT_BASE/rsna_${TASK}_${SUFFIX}"

    cat > "$CONFIG_PATH" <<YAML
model:
  model_name: "$PROCESSOR"
  subfolder: "$SUBFOLDER"
  num_classes: 3
  attention_cfg: null

data_dir: "$DATA_DIR"
val_split: 0.15
seed: 42

epochs: 25
batch_size: 512
learning_rate: 0.005
weight_decay: 0.0001
num_workers: 8
eval_steps: 4

dilation_radius: $DILATION
use_feature_caching: true

cache_suffix: "$SUFFIX"

task: "$TASK"
log_dir: "$ROOT/classification_hf/logs"
output_dir: "$OUTPUT_DIR"
use_pretrained_head: false
YAML

    echo "  Written: $CONFIG_PATH"
    CONFIGS+=("$CONFIG_PATH")
    OUTPUT_DIRS+=("$OUTPUT_DIR")
    mkdir -p "$OUTPUT_DIR"
done

echo ""
echo "Phase 3 complete."

# ── Phase 4 : Training runs ───────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "Phase 4: Training runs …"
echo "================================================================"

TOTAL_RUNS=$(( ${#CONFIGS[@]} * N_RUNS ))
echo "Total runs: $TOTAL_RUNS"

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
echo "Phase 4 complete."

# ── Phase 5 : Bootstrap analysis ─────────────────────────────────────────────
echo ""
echo "================================================================"
echo "Phase 5: Bootstrap analysis …"
echo "================================================================"

rm -f "$LOG_DIR/bootstrap_results.csv" "$LOG_DIR/bootstrap_pretrained_results.csv"

bootstrap_pids=()
for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    RUNS_DIR="${OUTPUT_DIRS[$i]}"

    echo ""
    echo "[$(date +%H:%M:%S)]  Bootstrap: task=$TASK  runs_dir=$RUNS_DIR"

    "$PYTHON" -m classification_hf.bootstrap_eval \
        --task        "$TASK" \
        --runs-dir    "$RUNS_DIR" \
        --n-bootstrap "$N_BOOTSTRAP" \
        --log-dir     "$LOG_DIR" &
    bootstrap_pids+=($!)
done

for pid in "${bootstrap_pids[@]}"; do
    wait "$pid"
done

echo ""
echo "================================================================"
echo "Bootstrap study complete — $(date)"
echo "Results:"
echo "  CSV  : $LOG_DIR/bootstrap_results.csv"
echo "  JSON : $LOG_DIR/bootstrap_{nfn,scs,ss}.json"
echo "================================================================"
