#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
TRAILING_INT_RE = re.compile(r"^(.*)__(\d+)$")


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def renumber_split(src_root: Path, dst_root: Path, split: str) -> None:
    src_img = src_root / "image" / split
    src_lbl = src_root / "label" / split
    dst_img = dst_root / "image" / split
    dst_lbl = dst_root / "label" / split

    imgs = list_images(src_img)
    if not imgs:
        print(f"[skip] no images in {src_img}")
        return

    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    for idx, img_fp in enumerate(imgs):
        lbl_fp = src_lbl / img_fp.name
        if not lbl_fp.exists():
            print(f"[skip] missing label for {img_fp.name}")
            continue

        stem = img_fp.stem
        m = TRAILING_INT_RE.match(stem)
        base = m.group(1) if m else stem
        new_name = f"{base}__{idx:08d}{img_fp.suffix.lower()}"

        out_img = dst_img / new_name
        out_lbl = dst_lbl / new_name

        if out_img.exists() or out_lbl.exists():
            raise RuntimeError(f"Collision detected for {new_name}")

        shutil.copy2(img_fp, out_img)
        shutil.copy2(lbl_fp, out_lbl)

    print(f"[ok] split={split} copied+renumbered {len(imgs)} pairs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", type=Path, required=True)
    ap.add_argument("--dst-root", type=Path, required=True)
    ap.add_argument("--splits", nargs="*", default=["train", "val"])
    args = ap.parse_args()

    for split in args.splits:
        renumber_split(args.src_root, args.dst_root, split)


if __name__ == "__main__":
    main()
