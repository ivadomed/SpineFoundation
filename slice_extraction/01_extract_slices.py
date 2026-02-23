#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm


def is_ignored(path: Path) -> bool:
    name = path.name.lower()
    if "preproc" in name or "cor" in name or "dwi" in name:
        return True
    return any(p.name.lower() in {"derivatives", ".git", "sourcedata"} for p in path.parents)


def iter_nii_gz(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.nii.gz"):
        if p.is_file() and not is_ignored(p):
            yield p


def load_ras(path: Path) -> nib.Nifti1Image:
    return nib.as_closest_canonical(nib.load(str(path)))


def get_spacing_ras(img: nib.Nifti1Image) -> Tuple[float, float, float]:
    dx, dy, dz = img.header.get_zooms()[:3]
    return float(dx), float(dy), float(dz)


def classify_from_spacing(spacing: Tuple[float, float, float], iso_tol: float, iso_eps_mm: float | None) -> str:
    dx, dy, dz = spacing
    smin, smax = min(dx, dy, dz), max(dx, dy, dz)
    if iso_eps_mm is not None:
        if (smax - smin) <= iso_eps_mm:
            return "isotropic"
    else:
        if (smax / smin) <= (1.0 + iso_tol):
            return "isotropic"
    if dz > dx and dz > dy:
        return "axial"
    if dx > dy and dx > dz:
        return "sagittal"
    return "axial"


def fmt_spacing(spacing_hw: Tuple[float, float]) -> str:
    a, b = spacing_hw
    ai = int(round(a * 1000))
    bi = int(round(b * 1000))
    return f"sp{ai:04d}x{bi:04d}"


def normalize_to_uint8(x: np.ndarray, pct: Tuple[float, float]) -> np.ndarray:
    x = np.asarray(x, np.float32)
    lo, hi = np.nanpercentile(x, pct)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=np.uint8)
    x = np.clip((x - lo) / (hi - lo), 0, 1)
    return (255 * x + 0.5).astype(np.uint8)


def labels_to_uint8(lbl: np.ndarray) -> np.ndarray:
    y = np.nan_to_num(lbl, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.rint(y)
    y = np.clip(y, 0, 255).astype(np.uint8)
    return y


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


def make_sliding_positions(full_size: int, tile_size: int, overlap: int) -> List[int]:
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
    tile_overlap: int,
    tile_threshold: int,
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    h, w = img_u8.shape
    must_tile = max(h, w) > tile_threshold

    if not must_tile:
        return [(0, img_u8, lbl_u8)]

    img_pad = pad_to_min_hw(img_u8, tile_size, tile_size, fill=0)
    lbl_pad = pad_to_min_hw(lbl_u8, tile_size, tile_size, fill=0)
    hp, wp = img_pad.shape

    xs = make_sliding_positions(wp, tile_size, tile_overlap)
    ys = make_sliding_positions(hp, tile_size, tile_overlap)

    out = []
    tid = 0
    for y0 in ys:
        for x0 in xs:
            it = img_pad[y0 : y0 + tile_size, x0 : x0 + tile_size]
            lt = lbl_pad[y0 : y0 + tile_size, x0 : x0 + tile_size]
            out.append((tid, it, lt))
            tid += 1
    return out


def extract_plane_slices(
    img_data: np.ndarray,
    lbl_data: np.ndarray,
    mode: str,
    spacing_ras: Tuple[float, float, float],
) -> List[Tuple[str, int, Tuple[float, float], np.ndarray, np.ndarray]]:
    if img_data.ndim == 4:
        img_data = img_data.mean(axis=-1)
    if lbl_data.ndim == 4:
        lbl_data = lbl_data[..., 0]

    dx, dy, dz = spacing_ras
    x, y, z = img_data.shape
    out: List[Tuple[str, int, Tuple[float, float], np.ndarray, np.ndarray]] = []

    if mode in ("axial", "isotropic"):
        spacing_hw = (dy, dx)
        for k in range(z):
            img_sl = np.transpose(img_data[:, :, k], (1, 0))
            lbl_sl = np.transpose(lbl_data[:, :, k], (1, 0))
            out.append(("axial", k, spacing_hw, img_sl, lbl_sl))

    if mode in ("sagittal", "isotropic"):
        spacing_hw = (dz, dy)
        for k in range(x):
            img_sl = np.transpose(img_data[k, :, :], (1, 0))
            lbl_sl = np.transpose(lbl_data[k, :, :], (1, 0))
            out.append(("sagittal", k, spacing_hw, img_sl, lbl_sl))

    return out


def deterministic_split(key: str, seed: int) -> float:
    h = hashlib.blake2b((str(seed) + key).encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / (2**64 - 1)


def sanitize_token(s: str) -> str:
    return "".join((c if (c.isalnum() or c in "._-") else "-") for c in s)


def find_label_match(img_path: Path, img_root: Path, label_root: Path, label_index: Dict[str, Path]) -> Path | None:
    rel = img_path.relative_to(img_root)
    direct = label_root / rel
    if direct.exists():
        return direct

    key = img_path.name.replace(".nii.gz", "")
    if key in label_index:
        return label_index[key]
    return None


@dataclass
class OutDirs:
    img_train: Path
    img_val: Path
    lbl_train: Path
    lbl_val: Path


def build_out_dirs(root: Path) -> OutDirs:
    dirs = OutDirs(
        img_train=root / "image" / "train",
        img_val=root / "image" / "val",
        lbl_train=root / "label" / "train",
        lbl_val=root / "label" / "val",
    )
    dirs.img_train.mkdir(parents=True, exist_ok=True)
    dirs.img_val.mkdir(parents=True, exist_ok=True)
    dirs.lbl_train.mkdir(parents=True, exist_ok=True)
    dirs.lbl_val.mkdir(parents=True, exist_ok=True)
    return dirs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-images", type=Path, required=True)
    ap.add_argument("--input-labels", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--train-ratio", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clip-pct", type=float, nargs=2, default=(0.5, 99.5))
    ap.add_argument("--iso-tol", type=float, default=0.1)
    ap.add_argument("--iso-eps-mm", type=float, default=None)
    ap.add_argument("--tile-size", type=int, default=224)
    ap.add_argument("--tile-overlap", type=int, default=56)
    ap.add_argument("--tile-threshold", type=int, default=512)
    args = ap.parse_args()

    if args.tile_overlap < 0 or args.tile_overlap >= args.tile_size:
        raise RuntimeError("tile-overlap must satisfy 0 <= tile-overlap < tile-size")

    image_root: Path = args.input_images
    label_root: Path = args.input_labels
    out_dirs = build_out_dirs(args.output_root)

    label_files = list(iter_nii_gz(label_root))
    label_index = {p.name.replace(".nii.gz", ""): p for p in label_files}

    extracted = skipped = errors = 0

    images = sorted(iter_nii_gz(image_root))
    for img_path in tqdm(images, desc="Extract", unit="vol"):
        try:
            lbl_path = find_label_match(img_path, image_root, label_root, label_index)
            if lbl_path is None:
                skipped += 1
                print(f"[SKIP] No matching label for {img_path}")
                continue

            img_nii = load_ras(img_path)
            lbl_nii = load_ras(lbl_path)
            spacing = get_spacing_ras(img_nii)

            img_data = img_nii.get_fdata(dtype=np.float32)
            lbl_data = lbl_nii.get_fdata(dtype=np.float32)

            if img_data.shape[:3] != lbl_data.shape[:3]:
                skipped += 1
                print(f"[SKIP] Shape mismatch image/label: {img_path.name}")
                continue

            mode = classify_from_spacing(spacing, args.iso_tol, args.iso_eps_mm)
            slices = extract_plane_slices(img_data, lbl_data, mode, spacing)

            src0 = sanitize_token(img_path.relative_to(image_root).parts[0]) if len(img_path.relative_to(image_root).parts) > 1 else "."
            base = sanitize_token(img_path.name.replace(".nii.gz", ""))

            for plane, sidx, spacing_hw, sl_img, sl_lbl in slices:
                img_u8 = normalize_to_uint8(sl_img, tuple(args.clip_pct))
                lbl_u8 = labels_to_uint8(sl_lbl)

                tiles = tile_pair(
                    img_u8=img_u8,
                    lbl_u8=lbl_u8,
                    tile_size=args.tile_size,
                    tile_overlap=args.tile_overlap,
                    tile_threshold=args.tile_threshold,
                )
                sp_tok = fmt_spacing(spacing_hw)

                for tidx, tile_img, tile_lbl in tiles:
                    fname = f"{src0}__{base}__{plane}__s{sidx:04d}__t{tidx:03d}__{sp_tok}.png"
                    split_train = deterministic_split(f"{base}__{plane}__s{sidx:04d}", args.seed) < args.train_ratio

                    img_out = out_dirs.img_train / fname if split_train else out_dirs.img_val / fname
                    lbl_out = out_dirs.lbl_train / fname if split_train else out_dirs.lbl_val / fname

                    Image.fromarray(tile_img, mode="L").save(img_out)
                    Image.fromarray(tile_lbl, mode="L").save(lbl_out)

            extracted += 1

        except Exception as exc:
            errors += 1
            print(f"[ERROR] {img_path.name} | {type(exc).__name__}: {exc}")
            traceback.print_exc()

    print(f"[SUMMARY] extracted={extracted} skipped={skipped} errors={errors}")


if __name__ == "__main__":
    main()
