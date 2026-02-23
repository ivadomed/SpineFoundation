#!/usr/bin/env python3
"""
numerotation_copy.py

Copy (do not rename) image files into another root folder while:
- preserving the same directory tree under train/ and val/
- adding or replacing a trailing '__<INT>' suffix (per directory), WITHOUT temp files

Safe under the assumption that original filenames do NOT already end with __<INT>
(or if they do, you want that suffix replaced).

Usage:
  python numerotation_copy.py --src-root /path/to/in_root --dst-root /path/to/out_root
"""

import argparse
import re
import shutil
from pathlib import Path
from typing import Iterable

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# CHANGED: match trailing "__<INT>"
TRAILING_INT_RE = re.compile(r"^(.*)__(\d+)$")


def iter_image_files(d: Path) -> Iterable[Path]:
    return sorted(
        p for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def renumber_copy_dir(src_dir: Path, dst_dir: Path) -> None:
    files = list(iter_image_files(src_dir))
    if not files:
        print(f"[skip] empty dir: {src_dir}")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    for i, src_path in enumerate(files):
        stem = src_path.stem

        # If the file already ends with "__<digits>", strip it
        m = TRAILING_INT_RE.match(stem)
        base = m.group(1) if m else stem

        # CHANGED: add "__<INT>" (two underscores)
        new_name = f"{base}__{i:08d}{src_path.suffix.lower()}"
        dst_path = dst_dir / new_name

        if dst_path.exists():
            raise RuntimeError(f"Collision detected: {dst_path}")

        shutil.copy2(src_path, dst_path)

    print(f"[ok] copied+renumbered {len(files)} files: {src_dir} -> {dst_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src-root",
        type=Path,
        required=True,
        help="Source root folder containing train/ and val/ (ImageNet-style)",
    )
    ap.add_argument(
        "--dst-root",
        type=Path,
        required=True,
        help="Destination root folder to write the renumbered copy",
    )
    args = ap.parse_args()

    src_root: Path = args.src_root
    dst_root: Path = args.dst_root

    if not src_root.is_dir():
        raise RuntimeError(f"--src-root does not exist or is not a directory: {src_root}")

    dst_root.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        src_split = src_root / split
        if not src_split.is_dir():
            print(f"[skip] missing split dir: {src_split}")
            continue

        # Mirror every subdirectory under split (keeps full arborescence)
        for src_dir in sorted(p for p in src_split.rglob("*") if p.is_dir()):
            rel = src_dir.relative_to(src_root)
            dst_dir = dst_root / rel
            renumber_copy_dir(src_dir, dst_dir)


if __name__ == "__main__":
    main()
