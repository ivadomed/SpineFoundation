#!/usr/bin/env bash
# extract_lesion_masks.sh
#
# Downloads ONLY the lesion label files (git annex get --include="*lesion*")
# for each MS/DCM dataset, then extracts them as PNG slices into
#   01_extracted_v2/label_lesion/train/<dataset>/
#
# Images are NOT re-downloaded (already in 01_extracted_v2/image/).
# The lesion PNGs will have the same filenames as the image PNGs so they can
# be matched directly by analyze_label_probing.py.
#
# Usage: bash extract_lesion_masks.sh [--dry-run]
set -uo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PYTHON=/home/ge.polymtl.ca/p123239/.conda/envs/dino/bin/python
EXTRACT_SCRIPT=/home/ge.polymtl.ca/p123239/FM/extract_lesion_slices.py
WORK_ROOT=/home/ge.polymtl.ca/p123239/data_work
CLONE_ROOT=$WORK_ROOT/00_cloned_lesion        # separate clone dir (not mixed with cord-seg clones)
OUT_ROOT=$WORK_ROOT/01_extracted_v2           # images already here; we add label_lesion/
LOG_DIR=$WORK_ROOT/logs_lesion
mkdir -p "$LOG_DIR" "$CLONE_ROOT"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# ── Dataset table  (url  label_dir_in_repo  label_suffix) ────────────────────
# Format: "URL|LABEL_SUBDIR|SUFFIX"
#   LABEL_SUBDIR: path inside the repo to the derivatives dir containing lesion masks
#   SUFFIX: the label suffix passed to 01_extract_slices.py
declare -A REPOS
REPOS=(
  ["canproco"]="git@data.neuro.polymtl.ca:datasets/canproco.git|derivatives/labels-ms-spinal-cord-only|_lesion-manual"
  ["nih-ms-mp2rage"]="git@data.neuro.polymtl.ca:datasets/nih-ms-mp2rage.git|derivatives/labels-ms-spinal-cord-only|_label-lesion_seg"
  # dcm-zurich-lesions : DCM, pas MS — exclus
  ["ms-basel-2018"]="git@data.neuro.polymtl.ca:datasets/ms-basel-2018.git|derivatives/labels|_lesion-manual"
  ["ms-nyu"]="git@data.neuro.polymtl.ca:datasets/ms-nyu.git|derivatives/labels|_lesion-manual"
  ["ms-karolinska-2020"]="git@data.neuro.polymtl.ca:datasets/ms-karolinska-2020.git|derivatives/labels|_lesion-manual"
  ["ms-multi-spine-challenge-2024"]="git@data.neuro.polymtl.ca:datasets/ms-multi-spine-challenge-2024.git|derivatives/labels|_label-lesion_seg"
  ["ms-mayo-critical-lesions-2025"]="git@data.neuro.polymtl.ca:datasets/ms-mayo-critical-lesions-2025.git|derivatives/labels|_label-criticalLesion_dseg"
  # msseg_challenge_2016 / msseg_challenge_2021 : lésions cérébrales, pas spinales — exclus
)

# ── Main loop ─────────────────────────────────────────────────────────────────
for NAME in "${!REPOS[@]}"; do
  IFS='|' read -r URL LABEL_SUBDIR SUFFIX <<< "${REPOS[$NAME]}"
  DEST="$CLONE_ROOT/$NAME"
  LOG="$LOG_DIR/lesion_${NAME}.log"
  LABEL_OUT="$OUT_ROOT/label_lesion"

  echo "================================================================"
  echo "[$(date +%H:%M:%S)] $NAME"
  echo "  url    : $URL"
  echo "  labels : $LABEL_SUBDIR"
  echo "  suffix : $SUFFIX"
  echo "================================================================"

  # ── Check if already done ─────────────────────────────────────────────────
  N_ALREADY=0
  for _d in "$LABEL_OUT/train/$NAME" "$LABEL_OUT/val/$NAME"; do
    [ -d "$_d" ] && N_ALREADY=$(( N_ALREADY + $(find "$_d" -name "*.png" | wc -l | tr -d ' ') ))
  done
  if [ "$N_ALREADY" -gt 0 ]; then
    echo "  $N_ALREADY lesion PNGs already extracted — skipping"
    continue
  fi

  if [ "$DRY_RUN" -eq 1 ]; then
    echo "  [DRY RUN] would clone + annex get lesion files + extract"
    continue
  fi

  # ── Clone (no annex get yet — just the git metadata) ─────────────────────
  if [ ! -d "$DEST/.git" ]; then
    echo "  git clone (metadata only)..."
    git clone "$URL" "$DEST" >> "$LOG" 2>&1
  else
    echo "  already cloned"
  fi

  # ── git annex get — lesion files only (much lighter than full annex get) ──
  LABEL_FULL="$DEST/$LABEL_SUBDIR"
  if [ -d "$LABEL_FULL" ]; then
    echo "  git annex get (all label nii.gz)..."
    pushd "$DEST" > /dev/null
    timeout 3600 git annex get \
      --jobs=4 \
      "$LABEL_SUBDIR/" >> "$LOG" 2>&1 || {
        echo "  WARNING: annex get returned non-zero (partial download?), continuing..."
      }
    popd > /dev/null
  else
    echo "  WARNING: $LABEL_SUBDIR not found in $DEST — skipping"
    continue
  fi

  # ── Count downloaded lesion files ─────────────────────────────────────────
  N_NII=0
  [ -d "$LABEL_FULL" ] && N_NII=$(find "$LABEL_FULL" \( -name "*lesion*.nii.gz" -o -name "*criticalLesion*.nii.gz" \) | wc -l | tr -d ' ')
  echo "  $N_NII lesion .nii.gz files downloaded"

  if [ "$N_NII" -eq 0 ]; then
    echo "  No lesion files found — skipping extraction"
    continue
  fi

  # ── Extract slices ────────────────────────────────────────────────────────
  # Reads existing image PNGs (already extracted) to know which slices to take,
  # then extracts only those slices from the lesion .nii.gz files.
  echo "  Extracting lesion label PNGs..."
  for SPLIT in train val; do
    IMG_DIR="$OUT_ROOT/image/$SPLIT/$NAME"
    OUT_DIR="$LABEL_OUT/$SPLIT/$NAME"
    [ -d "$IMG_DIR" ] || continue
    mkdir -p "$OUT_DIR"
    "$PYTHON" "$EXTRACT_SCRIPT" \
      --image_dir     "$IMG_DIR" \
      --label_nii_dir "$LABEL_FULL" \
      --label_suffix  "$SUFFIX" \
      --output_dir    "$OUT_DIR" \
      --workers       16 \
      >> "$LOG" 2>&1
  done

  N_OUT=0
  for _d in "$LABEL_OUT/train/$NAME" "$LABEL_OUT/val/$NAME"; do
    [ -d "$_d" ] && N_OUT=$(( N_OUT + $(find "$_d" -name "*.png" | wc -l | tr -d ' ') ))
  done
  echo "  Extracted $N_OUT lesion label PNGs"

  # ── Remove cloned repo (saves disk space) ─────────────────────────────────
  echo "  Removing cloned repo..."
  chmod -R u+w "$DEST" && rm -rf "$DEST"

  echo "  Done: $NAME"
  echo ""
done

echo "================================================================"
echo "Summary — lesion label PNGs in $OUT_ROOT/label_lesion/:"
for d in "$OUT_ROOT/label_lesion/train"/*/; do
  [ -d "$d" ] || continue
  n=$(find "$d" -name "*.png" | wc -l | tr -d ' ')
  [ "$n" -gt 0 ] && printf "  %-40s %d\n" "$(basename "$d")" "$n"
done
echo "================================================================"
