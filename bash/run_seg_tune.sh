#!/bin/bash
cd /home/ge.polymtl.ca/p123239/SpineFoundation

PYTHON_DINO=/home/ge.polymtl.ca/p123239/.conda/envs/dino/bin/python
PYTHON_FM=/home/ge.polymtl.ca/p123239/.conda/envs/FM/bin/python

CURIA=/home/ge.polymtl.ca/p123239/.cache/huggingface/hub/models--raidium--curia/snapshots/9657dc56276bc6c9503ef6f8d060879c8bee482f
DINOV3=/home/ge.polymtl.ca/p123239/FM/models/dinov3-vitl16
DINOV2REG=/home/ge.polymtl.ca/p123239/FM/models/dinov2-registers-large
BIOMEDCLIP=/home/ge.polymtl.ca/p123239/FM/models/biomedclip_hf
MRICORE=/home/ge.polymtl.ca/p123239/FM/models/mricore

DATA_CURIA=/home/ge.polymtl.ca/p123239/SpineFoundation/segmentation_hf/data
DATA_DINOV3=/home/ge.polymtl.ca/p123239/SpineFoundation/segmentation_hf/data/dinov3
DATA_DINOV2REG=/home/ge.polymtl.ca/p123239/SpineFoundation/segmentation_hf/data/dinov2reg
DATA_DINOV2REG_NPZ=/home/ge.polymtl.ca/p123239/SpineFoundation/segmentation_hf/data/brno_npz_dinov2reg
DATA_BIOMEDCLIP=/home/ge.polymtl.ca/p123239/SpineFoundation/segmentation_hf/data/biomedclip
DATA_BIOMEDCLIP_NPZ=/home/ge.polymtl.ca/p123239/SpineFoundation/segmentation_hf/data/brno_npz_biomedclip
DATA_MRICORE=/home/ge.polymtl.ca/p123239/SpineFoundation/segmentation_hf/data/mricore
DATA_MRICORE_NPZ=/home/ge.polymtl.ca/p123239/SpineFoundation/segmentation_hf/data/brno_npz_mricore

OUT_CURIA=/home/ge.polymtl.ca/p123239/SpineFoundation/outputs_seg
OUT_DINOV3=/home/ge.polymtl.ca/p123239/SpineFoundation/outputs_seg_dinov3
OUT_DINOV2REG=/home/ge.polymtl.ca/p123239/SpineFoundation/outputs_seg_dinov2reg
OUT_BIOMEDCLIP=/home/ge.polymtl.ca/p123239/SpineFoundation/outputs_seg_biomedclip
OUT_MRICORE=/home/ge.polymtl.ca/p123239/SpineFoundation/outputs_seg_mricore

LOG_DIR="$(dirname $0)/segmentation_hf/logs_seg_folds"
mkdir -p "$LOG_DIR"

# ── Curia ─────────────────────────────────────────────────────────────────────
echo "=== Curia ==="
for fold in 0 1 2 3 4 5 6 7 8 9 10 11; do
    DONE_MARKER="$OUT_CURIA/tune_fold${fold}/final/best.pt"
    if [ -f "$DONE_MARKER" ]; then
        echo "[Curia fold ${fold}] Déjà terminé, skip."
        continue
    fi

    echo "[Curia fold ${fold}] Lancement..."
    for attempt in 1 2 3; do
        $PYTHON_DINO -m segmentation_hf.tune \
            --model_dir $CURIA \
            --npz_train_dir $DATA_CURIA/fold_${fold}/train_npz \
            --npz_val_dir   $DATA_CURIA/fold_${fold}/val_npz \
            --output_dir    $OUT_CURIA/tune_fold${fold} \
            --n_trials 50 --trial_epochs 50 --final_epochs 300 --amp --batch_size 256 --image_size 512 \
            2>&1 | tee -a "$LOG_DIR/fold_${fold}.log" \
            && break \
            || echo "[Curia fold ${fold}] CRASH (tentative $attempt/3), retry dans 10s..."
        sleep 10
    done
    [ -f "$OUT_CURIA/tune_fold${fold}/final/best.pt" ] \
        && echo "[Curia fold ${fold}] Terminé." \
        || echo "[Curia fold ${fold}] ÉCHEC après 3 tentatives."
    sleep 5
done
echo "Tous les folds Curia terminés."

# ── Évaluation Curia sur test set ─────────────────────────────────────────────
echo "=== Évaluation Curia (test set) ==="
for fold in 0 1 2 3 4 5 6 7 8 9 10 11; do
    CKPT="$OUT_CURIA/tune_fold${fold}/final/best.pt"
    EVAL_OUT="$OUT_CURIA/tune_fold${fold}/test_eval"
    if [ ! -f "$CKPT" ]; then
        echo "[Curia eval fold ${fold}] Pas de checkpoint, skip."
        continue
    fi
    if [ -f "$EVAL_OUT/test_metrics_summary.json" ]; then
        echo "[Curia eval fold ${fold}] Déjà évalué, skip."
        continue
    fi
    echo "[Curia eval fold ${fold}] Évaluation..."
    $PYTHON_DINO -m segmentation_hf.evaluate_test \
        --checkpoint   $CKPT \
        --model_dir    $CURIA \
        --test_npz_dir $DATA_CURIA/test_npz \
        --output_dir   $EVAL_OUT \
        --image_size   512 \
        --amp \
        2>&1 | tee -a "$LOG_DIR/eval_fold_${fold}.log"
done
echo "Évaluations Curia terminées."

# ── DINOv3 ────────────────────────────────────────────────────────────────────
echo "=== DINOv3 ==="
for fold in 0 1 2 3 4 5 6 7 8 9 10 11; do
    DONE_MARKER="$OUT_DINOV3/tune_fold${fold}/final/best.pt"
    if [ -f "$DONE_MARKER" ]; then
        echo "[DINOv3 fold ${fold}] Déjà terminé, skip."
        continue
    fi

    echo "[DINOv3 fold ${fold}] Lancement..."
    for attempt in 1 2 3; do
        $PYTHON_FM -m segmentation_hf.tune \
            --model_dir $DINOV3 \
            --npz_train_dir $DATA_DINOV3/fold_${fold}/train_npz \
            --npz_val_dir   $DATA_DINOV3/fold_${fold}/val_npz \
            --output_dir    $OUT_DINOV3/tune_fold${fold} \
            --n_trials 50 --trial_epochs 50 --final_epochs 300 --amp --batch_size 256 \
            2>&1 | tee -a "$LOG_DIR/dinov3_fold_${fold}.log" \
            && break \
            || echo "[DINOv3 fold ${fold}] CRASH (tentative $attempt/3), retry dans 10s..."
        sleep 10
    done
    [ -f "$OUT_DINOV3/tune_fold${fold}/final/best.pt" ] \
        && echo "[DINOv3 fold ${fold}] Terminé." \
        || echo "[DINOv3 fold ${fold}] ÉCHEC après 3 tentatives."
    sleep 5
done
echo "Tous les folds DINOv3 terminés."

# ── Évaluation DINOv3 sur test set ────────────────────────────────────────────
echo "=== Évaluation DINOv3 (test set) ==="
for fold in 0 1 2 3 4 5 6 7 8 9 10 11; do
    CKPT="$OUT_DINOV3/tune_fold${fold}/final/best.pt"
    EVAL_OUT="$OUT_DINOV3/tune_fold${fold}/test_eval"
    if [ ! -f "$CKPT" ]; then
        echo "[DINOv3 eval fold ${fold}] Pas de checkpoint, skip."
        continue
    fi
    if [ -f "$EVAL_OUT/test_metrics_summary.json" ]; then
        echo "[DINOv3 eval fold ${fold}] Déjà évalué, skip."
        continue
    fi
    echo "[DINOv3 eval fold ${fold}] Évaluation..."
    $PYTHON_FM -m segmentation_hf.evaluate_test \
        --checkpoint   $CKPT \
        --model_dir    $DINOV3 \
        --test_npz_dir $DATA_DINOV3/test_npz \
        --output_dir   $EVAL_OUT \
        --amp \
        2>&1 | tee -a "$LOG_DIR/dinov3_eval_fold_${fold}.log"
done
echo "Évaluations DINOv3 terminées."

# ── DINOv2-registers : cache features ─────────────────────────────────────────
echo "=== DINOv2-registers : cache features ==="
if [ "$(ls -A $DATA_DINOV2REG_NPZ 2>/dev/null | wc -l)" -ge 5338 ]; then
    echo "Features déjà cachées, skip."
else
    echo "Extraction des patch tokens DINOv2-registers..."
    $PYTHON_FM -m segmentation_hf.cache_brno_features \
        --model_dir $DINOV2REG \
        --out_dir   $DATA_DINOV2REG_NPZ \
        2>&1 | tee -a "$LOG_DIR/dinov2reg_cache.log"
fi

# ── DINOv2-registers : symlinks ───────────────────────────────────────────────
echo "=== DINOv2-registers : symlinks ==="
if [ -d "$DATA_DINOV2REG/fold_0/train_npz" ]; then
    echo "Symlinks déjà créés, skip."
else
    $PYTHON_FM -m segmentation_hf.build_data_links \
        --npz_dir  $DATA_DINOV2REG_NPZ \
        --data_dir $DATA_DINOV2REG \
        2>&1 | tee -a "$LOG_DIR/dinov2reg_links.log"
fi

# ── DINOv2-registers : entraînement ───────────────────────────────────────────
echo "=== DINOv2-registers ==="
for fold in 0 1 2 3 4 5 6 7 8 9 10 11; do
    DONE_MARKER="$OUT_DINOV2REG/tune_fold${fold}/final/best.pt"
    if [ -f "$DONE_MARKER" ]; then
        echo "[DINOv2reg fold ${fold}] Déjà terminé, skip."
        continue
    fi

    echo "[DINOv2reg fold ${fold}] Lancement..."
    for attempt in 1 2 3; do
        $PYTHON_FM -m segmentation_hf.tune \
            --model_dir $DINOV2REG \
            --npz_train_dir $DATA_DINOV2REG/fold_${fold}/train_npz \
            --npz_val_dir   $DATA_DINOV2REG/fold_${fold}/val_npz \
            --output_dir    $OUT_DINOV2REG/tune_fold${fold} \
            --n_trials 50 --trial_epochs 50 --final_epochs 300 --amp --batch_size 256 \
            2>&1 | tee -a "$LOG_DIR/dinov2reg_fold_${fold}.log" \
            && break \
            || echo "[DINOv2reg fold ${fold}] CRASH (tentative $attempt/3), retry dans 10s..."
        sleep 10
    done
    [ -f "$OUT_DINOV2REG/tune_fold${fold}/final/best.pt" ] \
        && echo "[DINOv2reg fold ${fold}] Terminé." \
        || echo "[DINOv2reg fold ${fold}] ÉCHEC après 3 tentatives."
    sleep 5
done
echo "Tous les folds DINOv2-registers terminés."

# ── Évaluation DINOv2-registers sur test set ──────────────────────────────────
echo "=== Évaluation DINOv2-registers (test set) ==="
for fold in 0 1 2 3 4 5 6 7 8 9 10 11; do
    CKPT="$OUT_DINOV2REG/tune_fold${fold}/final/best.pt"
    EVAL_OUT="$OUT_DINOV2REG/tune_fold${fold}/test_eval"
    if [ ! -f "$CKPT" ]; then
        echo "[DINOv2reg eval fold ${fold}] Pas de checkpoint, skip."
        continue
    fi
    if [ -f "$EVAL_OUT/test_metrics_summary.json" ]; then
        echo "[DINOv2reg eval fold ${fold}] Déjà évalué, skip."
        continue
    fi
    echo "[DINOv2reg eval fold ${fold}] Évaluation..."
    $PYTHON_FM -m segmentation_hf.evaluate_test \
        --checkpoint   $CKPT \
        --model_dir    $DINOV2REG \
        --test_npz_dir $DATA_DINOV2REG/test_npz \
        --output_dir   $EVAL_OUT \
        --amp \
        2>&1 | tee -a "$LOG_DIR/dinov2reg_eval_fold_${fold}.log"
done
echo "Évaluations DINOv2-registers terminées."

# ── BiomedCLIP : cache features ───────────────────────────────────────────────
echo "=== BiomedCLIP : cache features ==="
if [ "$(ls -A $DATA_BIOMEDCLIP_NPZ 2>/dev/null | wc -l)" -ge 5338 ]; then
    echo "Features BiomedCLIP déjà cachées, skip."
else
    echo "Extraction des patch tokens BiomedCLIP..."
    $PYTHON_FM -m segmentation_hf.cache_brno_features \
        --model_dir $BIOMEDCLIP \
        --out_dir   $DATA_BIOMEDCLIP_NPZ \
        2>&1 | tee -a "$LOG_DIR/biomedclip_cache.log"
fi

# ── BiomedCLIP : symlinks ─────────────────────────────────────────────────────
echo "=== BiomedCLIP : symlinks ==="
if [ -d "$DATA_BIOMEDCLIP/fold_0/train_npz" ]; then
    echo "Symlinks BiomedCLIP déjà créés, skip."
else
    $PYTHON_FM -m segmentation_hf.build_data_links \
        --npz_dir  $DATA_BIOMEDCLIP_NPZ \
        --data_dir $DATA_BIOMEDCLIP \
        2>&1 | tee -a "$LOG_DIR/biomedclip_links.log"
fi

# ── BiomedCLIP : entraînement ─────────────────────────────────────────────────
echo "=== BiomedCLIP ==="
for fold in 0 1 2 3 4 5 6 7 8 9 10 11; do
    DONE_MARKER="$OUT_BIOMEDCLIP/tune_fold${fold}/final/best.pt"
    if [ -f "$DONE_MARKER" ]; then
        echo "[BiomedCLIP fold ${fold}] Déjà terminé, skip."
        continue
    fi
    echo "[BiomedCLIP fold ${fold}] Lancement..."
    for attempt in 1 2 3; do
        $PYTHON_FM -m segmentation_hf.tune \
            --model_dir $BIOMEDCLIP \
            --npz_train_dir $DATA_BIOMEDCLIP/fold_${fold}/train_npz \
            --npz_val_dir   $DATA_BIOMEDCLIP/fold_${fold}/val_npz \
            --output_dir    $OUT_BIOMEDCLIP/tune_fold${fold} \
            --n_trials 50 --trial_epochs 50 --final_epochs 300 --amp --batch_size 256 \
            2>&1 | tee -a "$LOG_DIR/biomedclip_fold_${fold}.log" \
            && break \
            || echo "[BiomedCLIP fold ${fold}] CRASH (tentative $attempt/3), retry dans 10s..."
        sleep 10
    done
    [ -f "$OUT_BIOMEDCLIP/tune_fold${fold}/final/best.pt" ] \
        && echo "[BiomedCLIP fold ${fold}] Terminé." \
        || echo "[BiomedCLIP fold ${fold}] ÉCHEC après 3 tentatives."
    sleep 5
done
echo "Tous les folds BiomedCLIP terminés."

# ── BiomedCLIP : évaluation test set ─────────────────────────────────────────
echo "=== Évaluation BiomedCLIP (test set) ==="
for fold in 0 1 2 3 4 5 6 7 8 9 10 11; do
    CKPT="$OUT_BIOMEDCLIP/tune_fold${fold}/final/best.pt"
    EVAL_OUT="$OUT_BIOMEDCLIP/tune_fold${fold}/test_eval"
    if [ ! -f "$CKPT" ]; then
        echo "[BiomedCLIP eval fold ${fold}] Pas de checkpoint, skip."
        continue
    fi
    if [ -f "$EVAL_OUT/test_metrics_summary.json" ]; then
        echo "[BiomedCLIP eval fold ${fold}] Déjà évalué, skip."
        continue
    fi
    echo "[BiomedCLIP eval fold ${fold}] Évaluation..."
    $PYTHON_FM -m segmentation_hf.evaluate_test \
        --checkpoint   $CKPT \
        --model_dir    $BIOMEDCLIP \
        --test_npz_dir $DATA_BIOMEDCLIP/test_npz \
        --output_dir   $EVAL_OUT \
        --amp \
        2>&1 | tee -a "$LOG_DIR/biomedclip_eval_fold_${fold}.log"
done
echo "Évaluations BiomedCLIP terminées."

# ── MRICore : cache features ──────────────────────────────────────────────────
echo "=== MRICore : cache features ==="
if [ "$(ls -A $DATA_MRICORE_NPZ 2>/dev/null | wc -l)" -ge 5338 ]; then
    echo "Features MRICore déjà cachées, skip."
else
    echo "Extraction des patch tokens MRICore..."
    $PYTHON_FM -m segmentation_hf.cache_brno_features_mricore \
        2>&1 | tee -a "$LOG_DIR/mricore_cache.log"
fi

# ── MRICore : symlinks ────────────────────────────────────────────────────────
echo "=== MRICore : symlinks ==="
if [ -d "$DATA_MRICORE/fold_0/train_npz" ]; then
    echo "Symlinks MRICore déjà créés, skip."
else
    $PYTHON_FM -m segmentation_hf.build_data_links \
        --npz_dir  $DATA_MRICORE_NPZ \
        --data_dir $DATA_MRICORE \
        2>&1 | tee -a "$LOG_DIR/mricore_links.log"
fi

# ── MRICore : entraînement ────────────────────────────────────────────────────
echo "=== MRICore ==="
for fold in 0 1 2 3 4 5 6 7 8 9 10 11; do
    DONE_MARKER="$OUT_MRICORE/tune_fold${fold}/final/best.pt"
    if [ -f "$DONE_MARKER" ]; then
        echo "[MRICore fold ${fold}] Déjà terminé, skip."
        continue
    fi
    echo "[MRICore fold ${fold}] Lancement..."
    for attempt in 1 2 3; do
        $PYTHON_FM -m segmentation_hf.tune \
            --model_dir $MRICORE \
            --in_channels 256 \
            --npz_train_dir $DATA_MRICORE/fold_${fold}/train_npz \
            --npz_val_dir   $DATA_MRICORE/fold_${fold}/val_npz \
            --output_dir    $OUT_MRICORE/tune_fold${fold} \
            --n_trials 50 --trial_epochs 50 --final_epochs 300 --amp --batch_size 128 --image_size 1024 \
            2>&1 | tee -a "$LOG_DIR/mricore_fold_${fold}.log" \
            && break \
            || echo "[MRICore fold ${fold}] CRASH (tentative $attempt/3), retry dans 10s..."
        sleep 10
    done
    [ -f "$OUT_MRICORE/tune_fold${fold}/final/best.pt" ] \
        && echo "[MRICore fold ${fold}] Terminé." \
        || echo "[MRICore fold ${fold}] ÉCHEC après 3 tentatives."
    sleep 5
done
echo "Tous les folds MRICore terminés."

# ── MRICore : évaluation test set ────────────────────────────────────────────
echo "=== Évaluation MRICore (test set) ==="
for fold in 0 1 2 3 4 5 6 7 8 9 10 11; do
    CKPT="$OUT_MRICORE/tune_fold${fold}/final/best.pt"
    EVAL_OUT="$OUT_MRICORE/tune_fold${fold}/test_eval"
    if [ ! -f "$CKPT" ]; then
        echo "[MRICore eval fold ${fold}] Pas de checkpoint, skip."
        continue
    fi
    if [ -f "$EVAL_OUT/test_metrics_summary.json" ]; then
        echo "[MRICore eval fold ${fold}] Déjà évalué, skip."
        continue
    fi
    echo "[MRICore eval fold ${fold}] Évaluation..."
    $PYTHON_FM -m segmentation_hf.evaluate_test \
        --checkpoint   $CKPT \
        --in_channels  256 \
        --test_npz_dir $DATA_MRICORE/test_npz \
        --output_dir   $EVAL_OUT \
        --image_size   1024 \
        --amp \
        2>&1 | tee -a "$LOG_DIR/mricore_eval_fold_${fold}.log"
done
echo "Évaluations MRICore terminées."

set_slot 1 CUDA_VISIBLE_DEVICES=1 bash /home/ge.polymtl.ca/p123239/SpineFoundation/run_scs_crop4cm.sh
