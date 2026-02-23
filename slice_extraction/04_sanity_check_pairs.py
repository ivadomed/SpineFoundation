#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image
from tqdm import tqdm


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
NAME_RE = re.compile(
    r"^.*__(sagittal|axial)__s(\d{3,6})__t(\d{3,6})__sp(\d{3,4})x(\d{3,4})(?:__(\d+))?$",
    re.IGNORECASE,
)


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--splits", nargs="*", default=["train", "val"])
    ap.add_argument("--max-issues", type=int, default=200)
    args = ap.parse_args()

    print("=== Sanity config ===")
    print(f"root       : {args.root}")
    print(f"splits     : {args.splits}")
    print(f"max-issues : {args.max_issues}")
    print("=====================")

    issues: list[str] = []

    def add_issue(msg: str) -> None:
        if len(issues) < args.max_issues:
            issues.append(msg)

    total = 0
    for split in args.splits:
        img_dir = args.root / "image" / split
        lbl_dir = args.root / "label" / split

        if not img_dir.exists():
            add_issue(f"[missing] image split folder: {img_dir}")
            continue
        if not lbl_dir.exists():
            add_issue(f"[missing] label split folder: {lbl_dir}")
            continue

        imgs = list_images(img_dir)
        if not imgs:
            add_issue(f"[empty] no files in {img_dir}")
            continue

        print(f"[info] split={split} files={len(imgs)}")

        pbar = tqdm(imgs, desc=f"Sanity {split}", unit="img", total=len(imgs))
        for img_fp in pbar:
            total += 1
            lbl_fp = lbl_dir / img_fp.name
            if not lbl_fp.exists():
                add_issue(f"[pair] missing label for {split}/{img_fp.name}")
                pbar.set_postfix(checked=total, issues=len(issues))
                continue

            if NAME_RE.match(img_fp.stem) is None:
                add_issue(f"[name] bad filename pattern: {img_fp.name}")

            try:
                with Image.open(img_fp) as i, Image.open(lbl_fp) as l:
                    if i.size != l.size:
                        add_issue(f"[shape] mismatch {split}/{img_fp.name}: image={i.size} label={l.size}")
            except Exception as exc:
                add_issue(f"[read] cannot read pair {split}/{img_fp.name}: {exc}")

            pbar.set_postfix(checked=total, issues=len(issues))

    print("=== SANITY SUMMARY ===")
    print(f"Root: {args.root}")
    print(f"Checked pairs: {total}")
    if issues:
        print(f"Issues: {len(issues)} (showing up to {args.max_issues})")
        for msg in issues:
            print(msg)
        return 2

    print("No issues found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
