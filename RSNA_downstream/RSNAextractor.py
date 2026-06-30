#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


SEVERITY_TO_CLASS = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}

TASK_CONFIG = {
    "nfn": {
        "modalities": ["T1w"],
        "vol_patterns": ["sub-*/anat/*_acq-sag_rec-*_T1w.nii.gz"],
        "label_glob": "_desc-*_label-*NeuralForaminalNarrowing_label.nii.gz",
        "slice_axis": 0,  # RAS axis 0 = R→L (sagittal slices)
    },
    "ss": {
        "modalities": ["T2w"],
        "vol_patterns": ["sub-*/anat/*_acq-ax_rec-*_T2w.nii.gz"],
        "label_glob": "_desc-*_label-*SubarticularStenosis_label.nii.gz",
        "slice_axis": 2,  # RAS axis 2 = I→S (axial slices)
    },
    "scs": {
        "modalities": ["T2w"],
        "vol_patterns": ["sub-*/anat/*_acq-sag_rec-*_T2w.nii.gz"],
        "label_glob": "_desc-*_label-SpinalCanalStenosis_label.nii.gz",
        "slice_axis": 0,  # RAS axis 0 = R→L (sagittal slices)
    },
}


def _load_json_sidecar(label_nii: Path) -> dict:
    p = Path(str(label_nii).replace("_label.nii.gz", "_label.json"))
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"No JSON sidecar found for label: {label_nii}")


def _pick_slice_with_positive(mask3d: np.ndarray, axis: int) -> int:
    """Return the index along `axis` that has the most positive voxels."""
    moved = np.moveaxis(mask3d, axis, 0)
    counts = moved.reshape(moved.shape[0], -1).sum(axis=1)
    if counts.max() <= 0:
        return -1
    return int(np.argmax(counts))


def _find_matching_labels(root: Path, vol_path: Path, task: str) -> list[Path]:
    cfg = TASK_CONFIG[task]
    base = vol_path.name

    matched_modality = None
    for modality in cfg["modalities"]:
        if base.endswith(f"_{modality}.nii.gz"):
            matched_modality = modality
            break
    if matched_modality is None:
        return []

    sub_id = vol_path.parent.parent.name
    labels_dir = root / "derivatives" / "labels" / sub_id / "anat"
    if not labels_dir.exists():
        return []

    suffix = f"_{matched_modality}.nii.gz"
    prefix = base.replace(suffix, f"_{matched_modality}")
    return sorted(labels_dir.glob(prefix + cfg["label_glob"]))


def _process_volume_group(job: tuple) -> list[str]:
    """
    Process one volume and all its associated label files.
    Loads the volume NIfTI once, then iterates over labels.
    Returns a list of error strings (empty = all OK).
    """
    vol_path, label_paths, out_dir, slice_axis = job
    errors = []

    # Load volume once for all labels
    try:
        img_ras = nib.as_closest_canonical(nib.load(str(vol_path)))
        img = img_ras.get_fdata(dtype=np.float32)
        zooms = img_ras.header.get_zooms()  # (through-plane, H, W) after canonical
    except Exception as e:
        return [f"[ERROR] load vol {vol_path.name}: {e}"]

    for label_path in label_paths:
        try:
            lab_ras = nib.as_closest_canonical(nib.load(str(label_path)))
            lab = lab_ras.get_fdata(dtype=np.float32)

            if img.shape != lab.shape:
                raise ValueError("Shape mismatch after RAS")

            mask = (lab > 0.5).astype(np.uint8)
            idx = _pick_slice_with_positive(mask, axis=slice_axis)
            if idx < 0:
                continue

            if slice_axis == 0:
                # Sagittal: (Y=A→P, Z=I→S) → transpose to (Z, Y)
                # In-plane axes are 1 (A→P) and 2 (I→S); after .T rows=Z=axis2, cols=Y=axis1
                img_2d  = np.ascontiguousarray(img[idx,  :, :].T)
                mask_2d = np.ascontiguousarray(mask[idx, :, :].T)
                spacing_mm = np.float32((float(zooms[1]) + float(zooms[2])) / 2.0)
            elif slice_axis == 2:
                # Axial: (X=R→L, Y=A→P) → transpose to (Y, X)
                # In-plane axes are 0 (R→L) and 1 (A→P); after .T rows=Y=axis1, cols=X=axis0
                img_2d  = np.ascontiguousarray(img[:,  :, idx].T)
                mask_2d = np.ascontiguousarray(mask[:, :, idx].T)
                spacing_mm = np.float32((float(zooms[0]) + float(zooms[1])) / 2.0)
            else:
                raise ValueError(f"Unsupported slice_axis={slice_axis}")

            js = _load_json_sidecar(label_path)
            sev = js.get("PathologySeverity")
            if sev not in SEVERITY_TO_CLASS:
                raise ValueError(f"Unknown PathologySeverity '{sev}'")

            cls = SEVERITY_TO_CLASS[sev]
            class_dir = out_dir / str(cls)
            class_dir.mkdir(parents=True, exist_ok=True)

            out_path = class_dir / label_path.name.replace(".nii.gz", ".npz")
            # savez (uncompressed) is much faster than savez_compressed
            np.savez(out_path, slice=img_2d, mask=mask_2d, spacing_mm=spacing_mm)

        except Exception as e:
            errors.append(f"[ERROR] {vol_path.name} + {label_path.name}: {e}")

    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",    type=str,
                    default="/home/ge.polymtl.ca/p123239/data_ok/lumbar-rsna-challenge-2024")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--task",    type=str, required=True, choices=["nfn", "ss", "scs"],
                    help="nfn: T1w NeuralForaminalNarrowing | "
                         "ss: T2w SubarticularStenosis | "
                         "scs: T2w SpinalCanalStenosis")
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                    help="Number of parallel workers (default: all CPUs)")
    args = ap.parse_args()

    root    = Path(args.root)
    out_dir = Path(args.out_dir)
    cfg     = TASK_CONFIG[args.task]

    vol_paths = sorted({
        p
        for pattern in cfg["vol_patterns"]
        for p in root.glob(pattern)
    })
    print(f"Found {len(vol_paths)} volumes for task '{args.task}'")

    # Build jobs: one job = (vol_path, [label_paths], out_dir, slice_axis)
    # Skip volumes with no matching labels upfront
    jobs = []
    for vol_path in vol_paths:
        labels = _find_matching_labels(root, vol_path, args.task)
        if labels:
            jobs.append((vol_path, labels, out_dir, cfg["slice_axis"]))

    print(f"{len(jobs)} volumes have labels — processing with {args.workers} workers")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process_volume_group, job): job[0] for job in jobs}
        with tqdm(total=len(jobs), desc="Volumes", unit="vol") as pbar:
            for future in as_completed(futures):
                for err in future.result():
                    print(err)
                pbar.update(1)


if __name__ == "__main__":
    main()
