"""
Crée l'arborescence de symlinks pour le fast path NPZ :
  data_dir/fold_N/{train_npz,val_npz}/  → npz_dir/*.npz
  data_dir/test_npz/                    → npz_dir/*.npz

Usage:
    # Curia (défaut)
    python -m segmentation_hf.build_data_links

    # DINOv3
    python -m segmentation_hf.build_data_links \
        --npz_dir  segmentation_hf/data/brno_npz_dinov3 \
        --data_dir segmentation_hf/data/dinov3
"""

import argparse
import json
from pathlib import Path

NNUNET_RAW  = Path("/home/ge.polymtl.ca/p123239/data/nnUNetv2/nnUNet_raw/Dataset026_BrnoSpineAll")
SPLITS_JSON = NNUNET_RAW / "splits_final.json"
TEST_JSON   = NNUNET_RAW / "test_set.json"

DEFAULT_DATA_DIR = Path(__file__).parent / "data"
DEFAULT_NPZ_DIR  = DEFAULT_DATA_DIR / "brno_npz"


def link(src: Path, dst: Path, dry_run: bool) -> bool:
    if not src.exists():
        print(f"  [WARN] NPZ introuvable : {src}")
        return False
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if not dry_run:
        dst.symlink_to(src)
    return True


def make_split_links(dst_dir: Path, identifiers: list[str], npz_dir: Path, dry_run: bool) -> int:
    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for ident in identifiers:
        src = npz_dir / f"{ident}.npz"
        dst = dst_dir / f"{ident}.npz"
        if link(src, dst, dry_run):
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--npz_dir",  default=None, help="Dossier des NPZ réels (défaut: data/brno_npz)")
    parser.add_argument("--data_dir", default=None, help="Dossier racine des folds (défaut: data/)")
    args = parser.parse_args()

    npz_dir  = Path(args.npz_dir).resolve()  if args.npz_dir  else DEFAULT_NPZ_DIR.resolve()
    data_dir = Path(args.data_dir).resolve() if args.data_dir else DEFAULT_DATA_DIR.resolve()

    splits = json.loads(SPLITS_JSON.read_text())
    test   = json.loads(TEST_JSON.read_text())

    tag = "[dry-run] " if args.dry_run else ""
    print(f"NPZ source : {npz_dir}")
    print(f"Data dir   : {data_dir}")

    for fold_idx, fold in enumerate(splits):
        fold_dir = data_dir / f"fold_{fold_idx}"
        print(f"\nFold {fold_idx}  (train={len(fold['train'])}, val={len(fold['val'])})")
        for split_name in ("train", "val"):
            dst_dir = fold_dir / f"{split_name}_npz"
            n = make_split_links(dst_dir, fold[split_name], npz_dir, args.dry_run)
            print(f"  {tag}{split_name}_npz : {n} symlinks")

    test_ids = test["images"]
    test_dir = data_dir / "test_npz"
    if not args.dry_run:
        test_dir.mkdir(parents=True, exist_ok=True)
    n_test = make_split_links(test_dir, test_ids, npz_dir, args.dry_run)
    print(f"\n{tag}test_npz : {n_test} symlinks")

    if args.dry_run:
        print("\n[dry-run] Rien créé.")
    else:
        print(f"\nSymlinks créés dans {data_dir}")


if __name__ == "__main__":
    main()
