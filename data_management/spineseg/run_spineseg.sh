#!/usr/bin/env bash
set -euo pipefail

# Run SCT spine segmentation (step1-only) for each image found under provided
# data root folders. For each image path matching */sub-*/anat/*.nii.gz the script
# will run:
#   sct_deepseg totalspineseg -i <image> -o <outdir> -step1-only 1
# where <outdir> = <root>/derivatives/labels/<sub-XXX>/anat
#
# Usage:
#   ./scripts/run_spineseg.sh [-n] [root_folder ...]
# Options:
#   -n  dry-run: only print commands without executing
# If no root_folder is provided, defaults to '/home/ge.polymtl.ca/p123239/data'.

DRY_RUN=0
while getopts ":n" opt; do
  case ${opt} in
    n ) DRY_RUN=1 ;;
    \? ) echo "Usage: $0 [-n] [root_folder ...]"; exit 1 ;;
  esac
done
shift $((OPTIND -1))

if [ $# -gt 0 ]; then
  ROOTS=("$@")
else
  ROOTS=("/home/ge.polymtl.ca/p123239/data")
fi

for root in "${ROOTS[@]}"; do
  echo "Processing root: $root"
  # find images under sub-*/**/anat/*.nii.gz
  while IFS= read -r img; do
    [ -z "$img" ] && continue
    # extract sub-XXX from path
    sub=$(echo "$img" | grep -o 'sub-[^/]*' | head -n1 || true)
    if [ -z "$sub" ]; then
      echo "  Warning: no sub-XXX found in path: $img -- skipping"
      continue
    fi

    outdir="$root/derivatives/labels/$sub/anat"
    mkdir -p "$outdir"

    cmd=(sct_deepseg totalspineseg -i "$img" -o "$outdir" -step1-only 1)
    echo "  Running: ${cmd[*]}"
    if [ "$DRY_RUN" -eq 0 ]; then
      "${cmd[@]}"
    fi
  done < <(find "$root" -type f -path "*/sub-*/anat/*.nii.gz" -iname "*.nii.gz" 2>/dev/null | sort)
done

echo "Done."
