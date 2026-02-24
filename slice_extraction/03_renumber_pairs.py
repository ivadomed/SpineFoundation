#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
TILE_TOKEN_RE = re.compile(r"__t\d{3,6}__", re.IGNORECASE)


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def overlap_pct_to_pixels(tile_size: int, overlap_pct: float) -> int:
    if overlap_pct < 0.0 or overlap_pct >= 100.0:
        raise RuntimeError("tile-overlap-pct must satisfy 0 <= pct < 100")
    overlap_px = int(round(tile_size * (overlap_pct / 100.0)))
    overlap_px = max(0, min(tile_size - 1, overlap_px))
    return overlap_px


def pad_to_min_hw(arr: np.ndarray, min_h: int, min_w: int, fill: int = 0) -> np.ndarray:
    h, w = arr.shape[:2]
    pad_h = max(0, min_h - h)
    pad_w = max(0, min_w - w)
    if pad_h == 0 and pad_w == 0:
        return arr

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return np.pad(arr, ((top, bottom), (left, right)), mode="constant", constant_values=fill)


def make_sliding_positions(full_size: int, tile_size: int, overlap: int) -> list[int]:
    stride = max(1, tile_size - overlap)
    if full_size <= tile_size:
        return [0]

    positions = list(range(0, full_size - tile_size + 1, stride))
    last = full_size - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def tile_pair(
    img_u8: np.ndarray,
    lbl_u8: np.ndarray,
    tile_size: int,
    tile_overlap_px: int,
    tile_threshold: int,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    h, w = img_u8.shape
    must_tile = max(h, w) > tile_threshold

    if not must_tile:
        return [(0, img_u8, lbl_u8)]

    img_pad = pad_to_min_hw(img_u8, tile_size, tile_size, fill=0)
    lbl_pad = pad_to_min_hw(lbl_u8, tile_size, tile_size, fill=0)
    hp, wp = img_pad.shape

    xs = make_sliding_positions(wp, tile_size, tile_overlap_px)
    ys = make_sliding_positions(hp, tile_size, tile_overlap_px)

    out = []
    tid = 0
    for y0 in ys:
        for x0 in xs:
            it = img_pad[y0 : y0 + tile_size, x0 : x0 + tile_size]
            lt = lbl_pad[y0 : y0 + tile_size, x0 : x0 + tile_size]
            out.append((tid, it, lt))
            tid += 1
    return out


def build_tile_stem(base_stem: str, tile_id: int) -> str:
    token = f"__t{tile_id:03d}__"
    if TILE_TOKEN_RE.search(base_stem):
        return TILE_TOKEN_RE.sub(token, base_stem)
    return f"{base_stem}{token}"


def renumber_split(
    src_root: Path,
    dst_root: Path,
    split: str,
    with_labels: bool,
    tiling: bool,
    tile_size: int,
    tile_overlap_pct: float,
    tile_threshold: int,
    skip_existing: bool,
) -> None:
    src_img = src_root / "image" / split
    src_lbl = src_root / "label" / split
    dst_img = dst_root / "image" / split
    dst_lbl = dst_root / "label" / split

    imgs = list_images(src_img)
    if not imgs:
        print(f"[skip] no images in {src_img}")
        return

    print(f"[info] split={split} images={len(imgs)}")
    tile_overlap_px = overlap_pct_to_pixels(tile_size=tile_size, overlap_pct=tile_overlap_pct)
    if tiling:
        print(
            f"[info] split={split} tiling=enabled | tile_size={tile_size}, "
            f"tile_overlap_pct={tile_overlap_pct}, tile_overlap_px={tile_overlap_px}, tile_threshold={tile_threshold}"
        )
    else:
        print(f"[info] split={split} tiling=disabled")

    dst_img.mkdir(parents=True, exist_ok=True)
    if with_labels:
        dst_lbl.mkdir(parents=True, exist_ok=True)

    written = 0
    missing = 0
    total_tiles = 0
    skipped_existing = 0
    pbar = tqdm(imgs, desc=f"Renumber {split}", unit="img", total=len(imgs))
    for img_fp in pbar:
        lbl_fp = src_lbl / img_fp.name
        if with_labels and not lbl_fp.exists():
            missing += 1
            pbar.set_postfix(written=written, missing=missing, tiles=total_tiles, skip_exist=skipped_existing)
            continue

        with Image.open(img_fp) as i:
            img_u8 = np.array(i.convert("L"), dtype=np.uint8)
        if with_labels:
            with Image.open(lbl_fp) as l:
                lbl_u8 = np.array(l.convert("L"), dtype=np.uint8)
        else:
            lbl_u8 = np.zeros_like(img_u8, dtype=np.uint8)

        if tiling:
            tiles = tile_pair(
                img_u8=img_u8,
                lbl_u8=lbl_u8,
                tile_size=tile_size,
                tile_overlap_px=tile_overlap_px,
                tile_threshold=tile_threshold,
            )
        else:
            tiles = [(0, img_u8, lbl_u8)]

        base = img_fp.stem

        for tile_id, tile_img, tile_lbl in tiles:
            tile_stem = build_tile_stem(base, tile_id)
            new_name = f"{tile_stem}{img_fp.suffix.lower()}"

            out_img = dst_img / new_name
            out_lbl = dst_lbl / new_name

            if with_labels:
                if skip_existing and out_img.exists() and out_lbl.exists():
                    skipped_existing += 1
                    continue
                if out_img.exists() or out_lbl.exists():
                    raise RuntimeError(f"Collision detected for {new_name}")
            else:
                if skip_existing and out_img.exists():
                    skipped_existing += 1
                    continue
                if out_img.exists():
                    raise RuntimeError(f"Collision detected for {new_name}")

            Image.fromarray(tile_img, mode="L").save(out_img)
            if with_labels:
                Image.fromarray(tile_lbl, mode="L").save(out_lbl)
            written += 1
            total_tiles += 1

        pbar.set_postfix(written=written, missing=missing, tiles=total_tiles, skip_exist=skipped_existing)

    if with_labels:
        print(
            f"[ok] split={split} written={written} missing_labels={missing} "
            f"tiles={total_tiles} skipped_existing={skipped_existing}"
        )
    else:
        print(f"[ok] split={split} written={written} tiles={total_tiles} skipped_existing={skipped_existing} mode=image-only")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", type=Path, required=True)
    ap.add_argument("--dst-root", type=Path, required=True)
    ap.add_argument("--splits", nargs="*", default=["train", "val"])
    ap.set_defaults(with_labels=True)
    ap.add_argument("--with-labels", dest="with_labels", action="store_true")
    ap.add_argument("--no-labels", dest="with_labels", action="store_false")
    ap.set_defaults(tiling=True)
    ap.add_argument("--tiling", dest="tiling", action="store_true")
    ap.add_argument("--no-tiling", dest="tiling", action="store_false")
    ap.add_argument("--tile-size", type=int, default=224)
    ap.add_argument("--tile-overlap-pct", type=float, default=25.0)
    ap.add_argument("--tile-threshold", type=int, default=512)
    ap.set_defaults(skip_existing=True)
    ap.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = ap.parse_args()

    print("=== Renumber config ===")
    print(f"src-root : {args.src_root}")
    print(f"dst-root : {args.dst_root}")
    print(f"splits   : {args.splits}")
    print(f"with-labels: {args.with_labels}")
    print(f"tiling   : {args.tiling}")
    print(f"tile-size: {args.tile_size}")
    print(f"overlap% : {args.tile_overlap_pct}")
    print(f"threshold: {args.tile_threshold}")
    print(f"skip-exist: {args.skip_existing}")
    print("=======================")

    for split in args.splits:
        renumber_split(
            src_root=args.src_root,
            dst_root=args.dst_root,
            split=split,
            with_labels=args.with_labels,
            tiling=args.tiling,
            tile_size=args.tile_size,
            tile_overlap_pct=args.tile_overlap_pct,
            tile_threshold=args.tile_threshold,
            skip_existing=args.skip_existing,
        )


if __name__ == "__main__":
    main()
