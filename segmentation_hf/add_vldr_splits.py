"""
Ajoute 3 folds "very low data regime" à splits_final.json (indices 9, 10, 11).
Utilise les slices avec foreground du sujet sub-3998B6406B (fold_0).

Tailles : 5 (4+1), 10 (8+2), 20 (16+4) — split 80/20, seed=42.

Usage:
    python -m segmentation_hf.add_vldr_splits [--dry-run]
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

NNUNET_RAW  = Path("/home/ge.polymtl.ca/p123239/data/nnUNetv2/nnUNet_raw/Dataset026_BrnoSpineAll")
SPLITS_JSON = NNUNET_RAW / "splits_final.json"
SEG_SPLITS  = Path("/home/ge.polymtl.ca/p123239/data/brno_seg_splits")

VLDR_SIZES = [5, 10, 20]
SEED       = 42
SUBJECT    = "sub-3998B6406B"


def has_foreground(mask_path: Path) -> bool:
    try:
        arr = np.array(Image.open(mask_path).convert("L"))
        return bool(arr.max() > 0)
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    splits = json.loads(SPLITS_JSON.read_text())

    # Collect all samples from fold_0 for SUBJECT
    fold0      = splits[0]
    all_ids    = sorted(set(fold0["train"]) | set(fold0["val"]))
    subj_ids   = [s for s in all_ids if s.startswith(SUBJECT)]

    # Filter to slices with foreground (check train_masks then val_masks)
    fg_ids = []
    for ident in sorted(subj_ids):
        for split_name in ("train", "val"):
            mask_dir = SEG_SPLITS / "fold_0" / f"{split_name}_masks"
            mask_path = mask_dir / f"{ident}_0000.png"
            if mask_path.exists():
                if has_foreground(mask_path):
                    fg_ids.append(ident)
                break

    print(f"Subject {SUBJECT}: {len(subj_ids)} slices total, {len(fg_ids)} avec foreground")

    # Check how many existing folds there are
    existing = len(splits)
    new_folds: list[dict] = []

    rng = random.Random(SEED)
    for size in VLDR_SIZES:
        if len(fg_ids) < size:
            print(f"[WARN] Pas assez de slices fg ({len(fg_ids)}) pour taille {size}, utilise toutes les slices fg")
            selected = list(fg_ids)
        else:
            # Take 'size' slices centered around the foreground region
            mid = len(fg_ids) // 2
            half = size // 2
            start = max(0, mid - half)
            end   = min(len(fg_ids), start + size)
            start = max(0, end - size)
            selected = fg_ids[start:end]

        # 80/20 split
        shuffled = list(selected)
        rng.shuffle(shuffled)
        n_val   = max(1, round(len(shuffled) * 0.2))
        n_train = len(shuffled) - n_val
        train_ids = shuffled[:n_train]
        val_ids   = shuffled[n_train:]

        new_folds.append({"train": sorted(train_ids), "val": sorted(val_ids)})
        fold_idx = existing + len(new_folds) - 1
        print(f"Fold {fold_idx} (VLDR-{size:02d}): {len(train_ids)} train + {len(val_ids)} val")

    if args.dry_run:
        print("[dry-run] Rien écrit.")
        return

    # Check if VLDR folds already exist (avoid duplicates)
    if len(splits) > 9:
        print(f"[WARN] splits_final.json a déjà {len(splits)} folds, les folds VLDR pourraient être dupliqués.")
        response = input("Continuer quand même ? [y/N] ").strip().lower()
        if response != "y":
            print("Annulé.")
            return

    splits.extend(new_folds)
    SPLITS_JSON.write_text(json.dumps(splits, indent=2))
    print(f"\nsplits_final.json mis à jour : {len(splits)} folds au total")
    print(f"Folds VLDR ajoutés : indices {existing} à {existing + len(new_folds) - 1}")


if __name__ == "__main__":
    main()
