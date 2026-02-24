#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm


def is_ignored(path: Path) -> bool:
    name = path.name.lower()
    if "preproc" in name or "cor" in name or "dwi" in name:
        return True
    return any(p.name.lower() in {"derivatives", ".git", "sourcedata"} for p in path.parents)


def iter_nii_gz(root: Path, apply_filters: bool = True) -> Iterable[Path]:
    for p in root.rglob("*.nii.gz"):
        if not p.is_file():
            continue
        if apply_filters and is_ignored(p):
            continue
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


def overlap_pct_to_pixels(tile_size: int, overlap_pct: float) -> int:
    if overlap_pct < 0.0 or overlap_pct >= 100.0:
        raise RuntimeError("tile-overlap-pct must satisfy 0 <= pct < 100")
    overlap_px = int(round(tile_size * (overlap_pct / 100.0)))
    overlap_px = max(0, min(tile_size - 1, overlap_px))
    return overlap_px


def tile_pair(
    img_u8: np.ndarray,
    lbl_u8: np.ndarray,
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    return [(0, img_u8, lbl_u8)]


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


def normalize_label_suffix(label_suffix: str) -> str:
    if label_suffix == "":
        return ""
    if label_suffix.startswith("_"):
        return label_suffix
    return f"_{label_suffix}"


def find_label_match(img_path: Path, img_root: Path, label_root: Path, label_suffix: str) -> Path | None:
    rel = img_path.relative_to(img_root)
    if rel.name.endswith(".nii.gz"):
        rel_seg = rel.with_name(rel.name.replace(".nii.gz", f"{label_suffix}.nii.gz"))
        direct_seg = label_root / rel_seg
        if direct_seg.exists():
            return direct_seg

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
    ap.add_argument("--input-labels", type=Path, default=None)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--train-ratio", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clip-pct", type=float, nargs=2, default=(0.5, 99.5))
    ap.add_argument("--iso-tol", type=float, default=0.1)
    ap.add_argument("--iso-eps-mm", type=float, default=None)
    ap.add_argument("--label-suffix", type=str, default="_seg", help="Suffix appended to image basename to find label (e.g. _seg, _spine)")
    ap.set_defaults(skip_existing=True)
    ap.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = ap.parse_args()
    args.label_suffix = normalize_label_suffix(args.label_suffix)
    with_labels = args.input_labels is not None

    print("=== Extract slices config ===")
    print(f"input-images     : {args.input_images}")
    print(f"input-labels     : {args.input_labels}")
    print(f"with-labels      : {with_labels}")
    print(f"output-root      : {args.output_root}")
    print(f"train-ratio      : {args.train_ratio}")
    print(f"seed             : {args.seed}")
    print(f"clip-pct         : {tuple(args.clip_pct)}")
    print(f"label-suffix     : {args.label_suffix}")
    print(f"skip-existing    : {args.skip_existing}")
    print("tiling           : disabled in stage 01 (moved after resampling)")
    print("=============================")

    image_root: Path = args.input_images
    label_root: Path | None = args.input_labels
    out_dirs = build_out_dirs(args.output_root)
    no_match_log_path = args.output_root / "no_matching_labels.txt"
    unmatched_labels_log_path = args.output_root / "labels_not_matched_to_images.txt"
    if with_labels:
        no_match_log_path.parent.mkdir(parents=True, exist_ok=True)
        no_match_log_path.write_text("")
        unmatched_labels_log_path.write_text("")

    images = sorted(iter_nii_gz(image_root, apply_filters=True))
    label_files: list[Path] = []
    if with_labels and label_root is not None:
        expected_label_files = []
        for img_path in images:
            rel = img_path.relative_to(image_root)
            if not rel.name.endswith(".nii.gz"):
                continue
            label_rel = rel.with_name(rel.name.replace(".nii.gz", f"{args.label_suffix}.nii.gz"))
            expected_label_files.append(label_root / label_rel)

        label_files = [p for p in expected_label_files if p.exists()]
    matched_labels: set[Path] = set()

    extracted = skipped = errors = 0
    total_slices = 0
    total_outputs = 0
    skipped_existing = 0

    if with_labels:
        print(f"[info] candidate images: {len(images)} | candidate labels: {len(label_files)}")
    else:
        print(f"[info] candidate images: {len(images)} | mode=image-only")
    pbar = tqdm(images, desc="Extract", unit="vol", total=len(images))
    for img_path in pbar:
        try:
            lbl_path = None
            if with_labels:
                if label_root is None:
                    raise RuntimeError("with_labels is enabled but input-labels is missing")
                lbl_path = find_label_match(img_path, image_root, label_root, args.label_suffix)
                if lbl_path is None:
                    skipped += 1
                    with no_match_log_path.open("a") as f:
                        f.write(str(img_path) + "\n")
                    pbar.set_postfix(extracted=extracted, skipped=skipped, errors=errors, outputs=total_outputs, skip_exist=skipped_existing)
                    continue
                matched_labels.add(lbl_path.resolve())

            img_nii = load_ras(img_path)
            spacing = get_spacing_ras(img_nii)

            img_data = img_nii.get_fdata(dtype=np.float32)
            if with_labels:
                if lbl_path is None:
                    raise RuntimeError("Label path unexpectedly missing")
                lbl_nii = load_ras(lbl_path)
                lbl_data = lbl_nii.get_fdata(dtype=np.float32)
            else:
                lbl_data = np.zeros_like(img_data, dtype=np.float32)

            if with_labels and img_data.shape[:3] != lbl_data.shape[:3]:
                skipped += 1
                print(f"[SKIP] Shape mismatch image/label: {img_path.name}")
                continue

            mode = classify_from_spacing(spacing, args.iso_tol, args.iso_eps_mm)
            slices = extract_plane_slices(img_data, lbl_data, mode, spacing)
            vol_outputs = 0

            src0 = sanitize_token(img_path.relative_to(image_root).parts[0]) if len(img_path.relative_to(image_root).parts) > 1 else "."
            base = sanitize_token(img_path.name.replace(".nii.gz", ""))

            for plane, sidx, spacing_hw, sl_img, sl_lbl in slices:
                total_slices += 1
                img_u8 = normalize_to_uint8(sl_img, tuple(args.clip_pct))
                lbl_u8 = labels_to_uint8(sl_lbl)

                tiles = tile_pair(
                    img_u8=img_u8,
                    lbl_u8=lbl_u8,
                )
                sp_tok = fmt_spacing(spacing_hw)

                for tidx, tile_img, tile_lbl in tiles:
                    vol_outputs += 1
                    total_outputs += 1
                    fname = f"{src0}__{base}__{plane}__s{sidx:04d}__t{tidx:03d}__{sp_tok}.png"
                    split_train = deterministic_split(f"{base}__{plane}__s{sidx:04d}", args.seed) < args.train_ratio

                    img_out = out_dirs.img_train / fname if split_train else out_dirs.img_val / fname
                    lbl_out = out_dirs.lbl_train / fname if split_train else out_dirs.lbl_val / fname

                    if with_labels:
                        if args.skip_existing and img_out.exists() and lbl_out.exists():
                            skipped_existing += 1
                            continue
                    else:
                        if args.skip_existing and img_out.exists():
                            skipped_existing += 1
                            continue

                    Image.fromarray(tile_img, mode="L").save(img_out)
                    if with_labels:
                        Image.fromarray(tile_lbl, mode="L").save(lbl_out)

            extracted += 1
            pbar.set_postfix(extracted=extracted, skipped=skipped, errors=errors, outputs=total_outputs, skip_exist=skipped_existing)

        except Exception as exc:
            errors += 1
            print(f"[ERROR] {img_path.name} | {type(exc).__name__}: {exc}")
            traceback.print_exc()
            pbar.set_postfix(extracted=extracted, skipped=skipped, errors=errors, outputs=total_outputs, skip_exist=skipped_existing)

    print(
        f"[SUMMARY] extracted={extracted} skipped={skipped} errors={errors} "
        f"total_slices={total_slices} total_outputs={total_outputs} skipped_existing={skipped_existing}"
    )
    if with_labels:
        print(f"[SUMMARY] no-matching-labels log: {no_match_log_path}")

        unmatched_labels = [p for p in label_files if p.resolve() not in matched_labels]
        if unmatched_labels:
            with unmatched_labels_log_path.open("w") as f:
                for p in unmatched_labels:
                    f.write(str(p) + "\n")
            print(f"[SUMMARY] labels not matched to images: {len(unmatched_labels)}")
            print(f"[SUMMARY] unmatched-labels log: {unmatched_labels_log_path}")
        else:
            print("[SUMMARY] labels not matched to images: 0")
    else:
        print("[SUMMARY] labels disabled: image-only extraction")


if __name__ == "__main__":
    main()
