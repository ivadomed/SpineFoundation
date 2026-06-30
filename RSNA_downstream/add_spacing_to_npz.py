#!/usr/bin/env python3
"""
Backfill 'spacing_mm' into existing NPZ patch files by reading the original NIfTI headers.

RSNAextractor.py now saves spacing_mm automatically, but files extracted before
this change must be patched with this script.

Usage:
    python -m RSNA_downstream.add_spacing_to_npz \
        --data_dir  /home/ge.polymtl.ca/p123239/data/RSNA_patches_scs \
        --nifti_root /home/ge.polymtl.ca/p123239/data_ok/lumbar-rsna-challenge-2024 \
        --task scs

The script resolves each NPZ filename back to its source NIfTI volume, reads the
voxel spacing from the header, and writes spacing_mm (float32 scalar) back into
the NPZ atomically.  Already-patched files are skipped unless --overwrite is passed.
"""

import argparse
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

# sub-XXXXXXXX_acq-sag_rec-XXXXXXXXXX_T2w_desc-... → (sub_id, rec_id, modality)
_RE_BIDS = re.compile(
    r"^(sub-\d+)_acq-(?:sag|ax)_rec-(\d+)_(T1w|T2w)_desc-"
)

TASK_SLICE_AXIS = {"nfn": 0, "scs": 0, "ss": 2}


def _nifti_spacing(vol_path: Path, slice_axis: int) -> float:
    """Return mean in-plane pixel spacing (mm) from a NIfTI header."""
    img = nib.as_closest_canonical(nib.load(str(vol_path)))
    zooms = img.header.get_zooms()
    if slice_axis == 0:
        # Sagittal: in-plane axes are 1 (A→P) and 2 (I→S)
        return float((zooms[1] + zooms[2]) / 2.0)
    elif slice_axis == 2:
        # Axial: in-plane axes are 0 (R→L) and 1 (A→P)
        return float((zooms[0] + zooms[1]) / 2.0)
    raise ValueError(f"Unsupported slice_axis={slice_axis}")


def _patch_npz(npz_path: str, spacing_mm: float) -> None:
    """Add spacing_mm to an NPZ file atomically."""
    d = np.load(npz_path)
    data = {k: d[k] for k in d.files}
    data["spacing_mm"] = np.float32(spacing_mm)
    dirpath = os.path.dirname(npz_path)
    fd, tmp = tempfile.mkstemp(dir=dirpath, suffix=".npz")
    os.close(fd)
    try:
        np.savez(tmp, **data)
        os.replace(tmp, npz_path)
    except Exception:
        os.unlink(tmp)
        raise


def _process(args: tuple) -> str | None:
    """Worker: resolve NIfTI path, read spacing, patch NPZ.  Returns error string or None."""
    npz_path, nifti_root, slice_axis, overwrite = args

    npz = Path(npz_path)

    # Skip if already patched
    if not overwrite:
        d = np.load(npz_path)
        if "spacing_mm" in d.files:
            return None

    m = _RE_BIDS.match(npz.name)
    if m is None:
        return f"[SKIP] cannot parse BIDS name: {npz.name}"

    sub_id, rec_id, modality = m.group(1), m.group(2), m.group(3)
    acq = "sag" if slice_axis == 0 else "ax"
    vol_glob = f"{sub_id}/anat/{sub_id}_acq-{acq}_rec-{rec_id}_{modality}.nii.gz"
    matches = list(nifti_root.glob(vol_glob))

    if not matches:
        return f"[WARN] NIfTI not found for {npz.name}  (glob: {vol_glob})"

    try:
        spacing = _nifti_spacing(matches[0], slice_axis)
        _patch_npz(npz_path, spacing)
    except Exception as e:
        return f"[ERROR] {npz.name}: {e}"

    return None


def main():
    ap = argparse.ArgumentParser(description="Backfill spacing_mm into RSNA patch NPZ files")
    ap.add_argument("--data_dir",    required=True,
                    help="Root patch directory (e.g. RSNA_patches_scs/) with 0/, 1/, 2/ sub-dirs")
    ap.add_argument("--nifti_root",  required=True,
                    help="Root of the BIDS NIfTI dataset (lumbar-rsna-challenge-2024/)")
    ap.add_argument("--task",        required=True, choices=["nfn", "scs", "ss"])
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--overwrite",   action="store_true",
                    help="Re-patch even files that already have spacing_mm")
    args = ap.parse_args()

    data_path   = Path(args.data_dir)
    nifti_root  = Path(args.nifti_root)
    slice_axis  = TASK_SLICE_AXIS[args.task]

    class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    all_npz = [str(f) for cd in class_dirs for f in sorted(cd.iterdir()) if f.suffix == ".npz"]
    print(f"Found {len(all_npz)} NPZ files in {data_path}")

    jobs = [(p, nifti_root, slice_axis, args.overwrite) for p in all_npz]
    errors = []

    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {ex.submit(_process, j): j[0] for j in jobs}
        for fut in tqdm(as_completed(futures), total=len(jobs),
                        desc="Patching NPZ", unit="file"):
            err = fut.result()
            if err:
                errors.append(err)

    print(f"\nDone. {len(errors)} error(s).")
    for e in errors[:20]:
        print(" ", e)


if __name__ == "__main__":
    main()
