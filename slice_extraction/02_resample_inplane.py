#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image
from tqdm import tqdm


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
RE_SP = re.compile(r"__sp(\d{3,4})x(\d{3,4})", re.IGNORECASE)


def fmt_sp(v_mm: float) -> str:
    return f"{int(round(v_mm * 1000)):04d}"


def pil_resample_mode(name: str) -> int:
    name = name.lower()
    if name == "nearest":
        return Image.Resampling.NEAREST
    if name == "bilinear":
        return Image.Resampling.BILINEAR
    if name == "bicubic":
        return Image.Resampling.BICUBIC
    if name == "lanczos":
        return Image.Resampling.LANCZOS
    raise ValueError(f"Unknown interp: {name}")


def list_split_files(base: Path, split: str) -> list[Path]:
    split_dir = base / split
    if not split_dir.is_dir():
        return []
    return sorted(
        p
        for p in split_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS and RE_SP.search(p.name)
    )


def process_pair(img_fp: Path, lbl_fp: Path, out_img: Path, out_lbl: Path, target: float, interp: str) -> None:
    m = RE_SP.search(img_fp.name)
    if m is None:
        return

    sp_h = int(m.group(1)) / 1000.0
    sp_w = int(m.group(2)) / 1000.0

    img = Image.open(img_fp).convert("L")
    lbl = Image.open(lbl_fp).convert("L")

    w, h = img.size
    new_h = max(1, int(round(h * (sp_h / target))))
    new_w = max(1, int(round(w * (sp_w / target))))

    img_rs = img.resize((new_w, new_h), resample=pil_resample_mode(interp))
    lbl_rs = lbl.resize((new_w, new_h), resample=Image.Resampling.NEAREST)

    target_token = f"__sp{fmt_sp(target)}x{fmt_sp(target)}"
    new_name = RE_SP.sub(target_token, img_fp.name)

    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_lbl.parent.mkdir(parents=True, exist_ok=True)

    img_rs.save(out_img.with_name(new_name))
    lbl_rs.save(out_lbl.with_name(new_name))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="Dataset root containing image/train,val and label/train,val")
    ap.add_argument("--target", type=float, default=0.8)
    ap.add_argument("--interp", type=str, default="bilinear", choices=["nearest", "bilinear", "bicubic", "lanczos"])
    ap.add_argument("--splits", nargs="*", default=["train", "val"])
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--inplace", action="store_true")
    ap.set_defaults(skip_existing=True)
    ap.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = ap.parse_args()

    print("=== Resample config ===")
    print(f"root      : {args.root}")
    print(f"target    : {args.target}")
    print(f"interp    : {args.interp}")
    print(f"splits    : {args.splits}")
    print(f"inplace   : {args.inplace}")
    print(f"out-root  : {args.out_root}")
    print(f"skip-exist: {args.skip_existing}")
    print("=======================")

    if not args.inplace and args.out_root is None:
        raise RuntimeError("Use --inplace or provide --out-root")

    image_root = args.root / "image"
    label_root = args.root / "label"
    if not image_root.exists() or not label_root.exists():
        raise RuntimeError("Expected folders: root/image and root/label")

    total_processed = 0
    total_missing = 0
    total_skipped_existing = 0

    for split in args.splits:
        files = list_split_files(image_root, split)
        print(f"[info] split={split} files={len(files)}")
        pbar = tqdm(files, desc=f"Resample {split}", unit="img", total=len(files))
        split_processed = 0
        split_missing = 0
        for img_fp in pbar:
            lbl_fp = label_root / split / img_fp.name
            if not lbl_fp.exists():
                split_missing += 1
                total_missing += 1
                pbar.set_postfix(done=split_processed, missing=split_missing)
                continue

            if args.inplace:
                out_img = img_fp
                out_lbl = lbl_fp
            else:
                out_img = args.out_root / "image" / split / img_fp.name
                out_lbl = args.out_root / "label" / split / lbl_fp.name

            m = RE_SP.search(img_fp.name)
            if m is None:
                continue
            target_token = f"__sp{fmt_sp(args.target)}x{fmt_sp(args.target)}"
            new_name = RE_SP.sub(target_token, img_fp.name)
            final_img = out_img.with_name(new_name)
            final_lbl = out_lbl.with_name(new_name)

            if args.skip_existing and final_img.exists() and final_lbl.exists():
                split_missing += 0
                total_skipped_existing += 1
                pbar.set_postfix(done=split_processed, missing=split_missing, skip_exist=total_skipped_existing)
                continue

            process_pair(img_fp, lbl_fp, out_img, out_lbl, args.target, args.interp)
            split_processed += 1
            total_processed += 1
            pbar.set_postfix(done=split_processed, missing=split_missing, skip_exist=total_skipped_existing)

            if args.inplace:
                m_old = RE_SP.search(img_fp.name)
                if m_old is not None:
                    target_token = f"__sp{fmt_sp(args.target)}x{fmt_sp(args.target)}"
                    new_name = RE_SP.sub(target_token, img_fp.name)
                    if new_name != img_fp.name:
                        img_old = img_fp
                        lbl_old = lbl_fp
                        if img_old.exists():
                            img_old.unlink()
                        if lbl_old.exists():
                            lbl_old.unlink()

        print(f"[ok] split={split} processed={split_processed} missing_labels={split_missing}")

    print(f"[SUMMARY] processed={total_processed} missing_labels={total_missing} skipped_existing={total_skipped_existing}")


if __name__ == "__main__":
    main()
