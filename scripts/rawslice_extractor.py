#!/usr/bin/env python3
"""
slice_extractor.py

Pipeline (updated):
1) Recursively scan --input-dir for *.nii.gz, excluding any under derivatives/ (and .git/)
   and files whose name contains "preproc" or "cor" or "dwi".
2) Load each NIfTI and reorient to RAS.
3) Classify acquisition from spacing (RAS):
   - axial: dz is largest
   - sagittal: dx is largest
   - isotropic: spacings approximately equal (ratio tol or abs eps)
4) Extract 2D slices:
   - axial: all XY slices (k over Z)
   - sagittal: all YZ slices (k over X)
   - isotropic: both axial + sagittal
5) Per-slice preprocessing:
   - robust intensity normalization to uint8 via percentile clipping
   - TEMP: no tiling (yields full slice)
6) STREAMING export to ImageNet folder structure on disk

Filename format:
<src0>__<base>__<plane>__sXXXX__tYY__spAAAAxBBBB.png

Where:
- src0 = first folder under --input-dir (or "." if file is directly under input-dir)
- base = original nii.gz basename without extension
- plane = axial/sagittal
- sXXXX = slice index
- tYY = tile index (always 0 for now)
- spAAAAxBBBB = in-plane spacing (H x W) in micrometers (mm*1000), e.g. 0.5mm -> 0500
"""

from __future__ import annotations

import argparse
import hashlib
import traceback
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import zoom as ndi_zoom


# ----------------------------- IO utils --------------------------------------

def is_under_derivatives(path: Path) -> bool:
    name = path.name.lower()
    if "preproc" in name or "cor" in name or "dwi" in name:
        return True
    return any(p.name.lower() in {"derivatives", ".git", "sourcedata"} for p in path.parents)


def iter_nii_gz(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.nii.gz"):
        if p.is_file() and not is_under_derivatives(p):
            yield p


def load_to_ras(path: Path) -> nib.Nifti1Image:
    return nib.as_closest_canonical(nib.load(str(path)))


def get_spacing_ras(img: nib.Nifti1Image) -> Tuple[float, float, float]:
    dx, dy, dz = img.header.get_zooms()[:3]
    return float(dx), float(dy), float(dz)


def first_level_dir(path: Path, root: Path) -> str:
    """
    Returns the first directory under `root` that contains `path`.
    If `path` is directly under `root`, returns ".".
    """
    try:
        rel = path.relative_to(root)
        return rel.parts[0] if len(rel.parts) > 1 else "."
    except ValueError:
        return "."


def sanitize_token(s: str) -> str:
    """
    Keep filenames safe: replace path separators/spaces with '-'.
    """
    return "".join((c if (c.isalnum() or c in "._-") else "-") for c in s)


# --------------------------- geometry -----------------------------------------

def classify_from_spacing(
    spacing: Tuple[float, float, float],
    iso_tol: float,
    iso_eps_mm: float | None,
) -> str:
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
    """
    Format spacing for filename suffix: spAAAAxBBBB
    spacing_hw is in mm. We store int(mm*1000) to avoid dots.
    """
    a, b = spacing_hw
    ai = int(round(a * 1000))
    bi = int(round(b * 1000))
    return f"sp{ai:04d}x{bi:04d}"


def extract_slices(
    img: nib.Nifti1Image,
    mode: str,
    spacing_ras: Tuple[float, float, float],
) -> List[Tuple[str, int, Tuple[float, float], np.ndarray]]:
    """
    Returns list of (plane, slice_index, spacing_hw, slice_2d)

    spacing_hw is the in-plane spacing (H, W) matching the returned slice array.

    - axial slice is (Y, X) => spacing_hw = (dy, dx)
    - sagittal slice is (Z, Y) => spacing_hw = (dz, dy)
    """
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data.mean(axis=-1)

    dx, dy, dz = spacing_ras
    X, Y, Z = data.shape
    out: List[Tuple[str, int, Tuple[float, float], np.ndarray]] = []

    if mode in ("axial", "isotropic"):
        spacing_hw = (dy, dx)  # (H=Y, W=X)
        for k in range(Z):
            sl = np.transpose(data[:, :, k], (1, 0))  # (Y, X)
            out.append(("axial", k, spacing_hw, sl))

    if mode in ("sagittal", "isotropic"):
        spacing_hw = (dz, dy)  # (H=Z, W=Y)
        for k in range(X):
            sl = np.transpose(data[k, :, :], (1, 0))  # (Z, Y)
            out.append(("sagittal", k, spacing_hw, sl))

    return out


# ------------------------- cropping -------------------------------------------

def crop_nonzero_border(img: np.ndarray, eps: float = 100.0) -> np.ndarray:
    """
    Crop black borders (rows/cols that are entirely zero or <= eps).
    """
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got {img.shape}")

    mask = np.abs(img) > eps
    if not mask.any():
        return img

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]

    y0, y1 = rows[0], rows[-1] + 1
    x0, x1 = cols[0], cols[-1] + 1

    return img[y0:y1, x0:x1]


# -------------------------- preprocessing ------------------------------------

def normalize_to_uint8(x: np.ndarray, pct: Tuple[float, float]) -> np.ndarray:
    x = np.asarray(x, np.float32)
    lo, hi = np.nanpercentile(x, pct)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=np.uint8)
    x = np.clip((x - lo) / (hi - lo), 0, 1)
    return (255 * x + 0.5).astype(np.uint8)


def resample_inplane_u8(
    img: np.ndarray,
    spacing_hw: Tuple[float, float],
    target: Tuple[float, float] = (1, 1),
    order: int = 1,
) -> np.ndarray:
    sy, sx = spacing_hw
    ty, tx = target
    zy, zx = sy / ty, sx / tx
    if abs(zy - 1) < 1e-6 and abs(zx - 1) < 1e-6:
        return img
    y = ndi_zoom(img.astype(np.float32), (zy, zx), order=order, mode="nearest")
    return np.clip(y + 0.5, 0, 255).astype(np.uint8)


def iter_slices_height_near_square(img: np.ndarray, tol: float = 0.10):
    """
    TEMP: no tiling at all. Always yield the full image as a single tile.
    """
    yield 0, img


def deterministic_split(key: str, seed: int) -> float:
    h = hashlib.blake2b((str(seed) + key).encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / (2**64 - 1)


def already_extracted(prefix: str, train_dir: Path, val_dir: Path) -> bool:
    """
    Skip if at least one output file with this prefix already exists.
    """
    for d in (train_dir, val_dir):
        if d.exists() and any(p.name.startswith(prefix) for p in d.iterdir()):
            return True
    return False


# ------------------------------- main -----------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="data_ok/")
    ap.add_argument("--output-root", default="slice_extractor/out_raw")
    ap.add_argument("--train-ratio", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clip-pct", type=float, nargs=2, default=(0.5, 99.5))
    ap.add_argument("--iso-tol", type=float, default=0.1)
    ap.add_argument("--iso-eps-mm", type=float, default=None)
    ap.add_argument("--class-folder", default="n00000000")
    ap.add_argument("--resample-order", type=int, default=1)
    args = ap.parse_args()

    input_root = Path(args.input_dir)

    root = Path(args.output_root)
    train_dir = root / "train" / args.class_folder
    val_dir = root / "val" / args.class_folder
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    (root / "labels.txt").write_text(args.class_folder + "\n")

    skipped = extracted = errors = 0

    for p in tqdm(sorted(iter_nii_gz(input_root)), total=None):
        t_file_start = time.perf_counter()

        base = p.name.replace(".nii.gz", "")
        src0 = sanitize_token(first_level_dir(p, input_root))
        file_prefix = f"{src0}__{base}__"

        if already_extracted(file_prefix, train_dir, val_dir):
            skipped += 1
            print(f"[SKIP] {file_prefix}*")
            continue

        try:
            # --- load + RAS ---
            img = load_to_ras(p)
            spacing = get_spacing_ras(img)

            # --- classify ---
            mode = classify_from_spacing(spacing, args.iso_tol, args.iso_eps_mm)

            # --- extract slices (also returns in-plane spacing) ---
            slices = extract_slices(img, mode, spacing)

            for plane, sidx, spacing_hw, sl in slices:
                # --- preprocess ---
                sl_c = crop_nonzero_border(sl, eps=0.0)
                u8 = normalize_to_uint8(sl_c, args.clip_pct)
                if u8.size == 0:
                    continue

                # --- save ---
                tidx = 0
                sp_tok = fmt_spacing(spacing_hw)
                fname = f"{src0}__{base}__{plane}__s{sidx:04d}__{sp_tok}.png"

                out = (
                    train_dir
                    if deterministic_split(fname, args.seed) < args.train_ratio
                    else val_dir
                )
                Image.fromarray(u8, mode="L").save(out / fname)

            extracted += 1
            _ = time.perf_counter() - t_file_start

        except Exception as e:
            errors += 1
            print(f"[ERROR] {file_prefix} | {type(e).__name__}: {e}")
            traceback.print_exc()

    print(f"\n[SUMMARY] extracted={extracted} skipped={skipped} errors={errors}")


if __name__ == "__main__":
    main()
