#!/usr/bin/env python3
"""
inplane_resample_slices.py

In-plane resample 2D slices to isotropic spacing using spacing encoded
in filename (__spXXXXxXXXX), following your axial/sagittal conventions.
"""

import argparse
import re
from pathlib import Path
from PIL import Image
from tqdm import tqdm

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

RE_SP = re.compile(r"__sp(\d{3,4})x(\d{3,4})", re.IGNORECASE)


def fmt_sp(v_mm: float) -> str:
    return f"{int(round(v_mm * 100)):04d}"


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--target", type=float, default=0.8)
    ap.add_argument("--interp", type=str, default="bilinear",
                    choices=["nearest", "bilinear", "bicubic", "lanczos"])
    ap.add_argument("--splits", nargs="*", default=["train", "val"])
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--inplace", action="store_true", help="Modify files in-place")
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--verbose", action="store_true", help="Print every processed file")
    args = ap.parse_args()

    root: Path = args.root
    if not root.is_dir():
        raise RuntimeError(f"Root not found: {root}")
    args.inplace=False
    if not args.inplace and args.out_root is None:
        raise RuntimeError("Use --inplace or provide --out-root")

    target = float(args.target)
    target_token = f"__sp{fmt_sp(target)}x{fmt_sp(target)}"
    resample = pil_resample_mode(args.interp)

    print("=== In-plane isotropic resampling ===")
    print(f"Root          : {root}")
    print(f"Splits        : {args.splits}")
    print(f"Target spacing: {target} mm")
    print(f"Interpolation : {args.interp}")
    print(f"In-place      : {args.inplace}")
    print(f"Skip existing : {args.skip_existing}")
    print("===================================")

    files = []
    for split in args.splits:
        d = root / split
        if not d.is_dir():
            continue
        files.extend(
            p for p in d.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXTS and RE_SP.search(p.name)
        )

    print(f"[info] Found {len(files)} files with spacing token")

    for fp in tqdm(files, desc="Resampling", unit="img"):
        name = fp.name
        m = RE_SP.search(name)
        if not m:
            continue

        sp_h = int(m.group(1)) / 1000.0
        sp_w = int(m.group(2)) / 1000.0

        if args.skip_existing and target_token in name:
            continue

        img = Image.open(fp).convert("L")
        W, H = img.size

        new_H = max(1, int(round(H * (sp_h / target))))
        new_W = max(1, int(round(W * (sp_w / target))))

        if new_H == H and new_W == W and target_token in name:
            continue

        img2 = img.resize((new_W, new_H), resample=resample)

        new_name = RE_SP.sub(target_token, name)

        if args.inplace:
            out_fp = fp.with_name(new_name)
        else:
            rel = fp.relative_to(root)
            out_fp = args.out_root / rel
            out_fp = out_fp.with_name(new_name)
            out_fp.parent.mkdir(parents=True, exist_ok=True)

        if out_fp.exists() and out_fp.resolve() != fp.resolve():
            raise RuntimeError(f"Collision: {out_fp}")

        img2.save(out_fp)

        if args.inplace and out_fp.name != fp.name:
            fp.unlink()

        if args.verbose:
            print(
                f"[ok] {fp.name} | "
                f"sp=({sp_h:.3f},{sp_w:.3f}) → ({target:.3f},{target:.3f}) | "
                f"({H},{W}) → ({new_H},{new_W})"
            )

    print("[done] Resampling completed")


if __name__ == "__main__":
    main()
