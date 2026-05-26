"""
extract_lesion_slices.py

For each image PNG in 01_extracted_v2/image/train/<dataset>/,
finds the matching lesion .nii.gz (downloaded via git annex get),
extracts the corresponding 2-D slice, and saves it as PNG to
01_extracted_v2/label_lesion/train/<dataset>/.

No image volumes need to be downloaded — only lesion .nii.gz files.
"""
from __future__ import annotations

import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm

PLANE_AXIS = {"axial": 2, "sagittal": 0, "coronal": 1}

# Populated once in main(), read by workers via module global
_NII_INDEX: dict[str, Path] = {}
_OUTPUT_DIR: Path = Path(".")


def parse_png_name(name: str) -> dict | None:
    stem = Path(name).stem
    parts = stem.split("__")
    if len(parts) < 5:
        return None
    subject = parts[0]   # e.g. "sub-nih001"
    plane   = parts[2]
    m_s     = re.match(r"s(\d+)", parts[3])
    m_t     = re.match(r"t(\d+)", parts[4])
    if not m_s or not m_t or plane not in PLANE_AXIS:
        return None
    return dict(subject=subject, plane=plane, sidx=int(m_s.group(1)), tidx=int(m_t.group(1)))


def extract_slice(nii_path: Path, plane: str, sidx: int, tidx: int) -> np.ndarray | None:
    try:
        img  = nib.as_closest_canonical(nib.load(str(nii_path)))
        data = np.asarray(img.dataobj)
    except Exception:
        return None
    if data.ndim == 4:
        data = data[..., min(tidx, data.shape[3] - 1)]
    if data.ndim != 3:
        return None
    axis = PLANE_AXIS[plane]
    if sidx >= data.shape[axis]:
        return None
    sl = np.take(data, sidx, axis=axis)
    return ((sl > 0).astype(np.uint8)) * 255


def _worker(png_name: str) -> str:
    """Module-level worker — pickling-safe for ProcessPoolExecutor."""
    out_path = _OUTPUT_DIR / png_name
    if out_path.exists():
        return "skip"
    info = parse_png_name(png_name)
    if info is None:
        return "parse_error"
    nii_path = _NII_INDEX.get(info["subject"])
    if nii_path is None:
        return "no_label"
    arr = extract_slice(nii_path, info["plane"], info["sidx"], info["tidx"])
    if arr is None:
        return "extract_error"
    Image.fromarray(arr, mode="L").save(out_path)
    return "ok"


def main() -> None:
    global _NII_INDEX, _OUTPUT_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir",     required=True)
    parser.add_argument("--label_nii_dir", required=True)
    parser.add_argument("--label_suffix",  required=True)
    parser.add_argument("--output_dir",    required=True)
    parser.add_argument("--workers",       type=int, default=8)
    args = parser.parse_args()

    image_dir     = Path(args.image_dir)
    label_nii_dir = Path(args.label_nii_dir)
    _OUTPUT_DIR   = Path(args.output_dir)
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    png_files = sorted(image_dir.glob("*.png"))
    print(f"Image PNGs      : {len(png_files):,}")
    print(f"Label nii dir   : {label_nii_dir}")
    print(f"Suffix          : {args.label_suffix}")
    print(f"Output          : {_OUTPUT_DIR}")

    # Build index: subject_id → nii path (matches "sub-XXX" prefix of BIDS filenames)
    suffix = args.label_suffix
    _NII_INDEX = {}
    for p in label_nii_dir.rglob(f"*{suffix}.nii.gz"):
        m = re.match(r"(sub-[^_]+)", p.name)
        if m:
            _NII_INDEX[m.group(1)] = p  # last file wins if multiple per subject
    print(f"Lesion .nii.gz  : {len(_NII_INDEX):,}")

    if not _NII_INDEX:
        # Debug: show what IS there
        all_nii = list(label_nii_dir.rglob("*.nii.gz"))
        print(f"  (all .nii.gz in dir: {len(all_nii)} — first 5 below)")
        for p in all_nii[:5]:
            print(f"    {p.name}")
        return

    # Pre-filter: only submit PNGs whose base subject is in the NII index.
    # Avoids submitting 400k+ "no_label" tasks when only a few subjects have masks.
    matched_files = []
    skipped_no_label = 0
    for p in png_files:
        info = parse_png_name(p.name)
        if info is None or info["subject"] not in _NII_INDEX:
            skipped_no_label += 1
        else:
            matched_files.append(p)
    print(f"Pre-filtered     : {len(matched_files):,} PNGs with matching label "
          f"({skipped_no_label:,} skipped, no label for subject)")

    counts: dict[str, int] = {"no_label": skipped_no_label}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_worker, p.name): p for p in matched_files}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Extracting"):
            r = fut.result()
            counts[r] = counts.get(r, 0) + 1

    print("\n── Summary ──────────────────────────────────────────────────")
    for k, v in sorted(counts.items()):
        print(f"  {k:15s}: {v:,}")
    n_out = len(list(_OUTPUT_DIR.glob("*.png")))
    print(f"  output PNGs    : {n_out:,}")


if __name__ == "__main__":
    main()
