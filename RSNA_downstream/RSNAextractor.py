#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import nibabel as nib


SEVERITY_TO_CLASS = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
LABEL_SUFFIX = "_label.nii.gz"

TASK_CONFIG = {
    "nfn": {
        "modality": "T1w",
        "vol_pattern": "sub-*/anat/*_acq-sag_rec-*_T1w.nii.gz",
        "label_glob": "_desc-*_label-*NeuralForaminalNarrowing_label.nii.gz",
        "slice_axis": 0,  # RAS axis 0 = R→L (sagittal slices)
    },
    "ss": {
        "modality": "T2w",
        "vol_pattern": "sub-*/anat/*_acq-ax_rec-*_T2w.nii.gz",
        "label_glob": "_desc-*_label-*SubarticularStenosis_label.nii.gz",
        "slice_axis": 2,  # RAS axis 2 = I→S (axial slices)
    },
    "scs": {
        "modality": "T2w",
        "vol_pattern": "sub-*/anat/*_acq-sag_rec-*_T2w.nii.gz",
        "label_glob": "_desc-*_label-SpinalCanalStenosis_label.nii.gz",
        "slice_axis": 0,  # RAS axis 0 = R→L (sagittal slices)
    },
}


def load_json_sidecar(label_nii: Path) -> dict:
    p = Path(str(label_nii).replace("_label.nii.gz", "_label.json"))
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"No JSON sidecar found for label: {label_nii}")


def as_ras(img: nib.Nifti1Image) -> nib.Nifti1Image:
    return nib.as_closest_canonical(img)


def pick_slice_with_positive(mask3d: np.ndarray, axis: int) -> int:
    """Return the index along `axis` that has the most positive voxels."""
    # Move target axis to front, then sum over remaining axes
    moved = np.moveaxis(mask3d, axis, 0)
    counts = moved.reshape(moved.shape[0], -1).sum(axis=1)
    if counts.max() <= 0:
        return -1
    return int(np.argmax(counts))


def crop_around_positive(img2d: np.ndarray, mask2d: np.ndarray, size: int = 200) -> np.ndarray:
    ys, zs = np.where(mask2d > 0)
    if len(ys) == 0:
        cy, cz = img2d.shape[0] // 2, img2d.shape[1] // 2
    else:
        cy, cz = int(ys[0]), int(zs[0])

    half = size // 2
    H, W = img2d.shape

    y0 = min(max(cy - half, 0), H - size)
    z0 = min(max(cz - half, 0), W - size)
    y0 = max(y0, 0)
    z0 = max(z0, 0)

    patch = img2d[y0:y0 + size, z0:z0 + size]

    if patch.shape != (size, size):
        padded = np.zeros((size, size), dtype=img2d.dtype)
        padded[:patch.shape[0], :patch.shape[1]] = patch
        return padded

    return patch


def find_matching_labels(root: Path, vol_path: Path, task: str):
    cfg = TASK_CONFIG[task]
    modality = cfg["modality"]
    suffix = f"_{modality}.nii.gz"

    base = vol_path.name
    if not base.endswith(suffix):
        return []

    sub_id = vol_path.parent.parent.name
    labels_dir = root / "derivatives" / "labels" / sub_id / "anat"
    if not labels_dir.exists():
        return []

    prefix = base.replace(suffix, f"_{modality}")
    return sorted(labels_dir.glob(prefix + cfg["label_glob"]))


def extract_and_save_patch(
    vol_path: Path,
    label_path: Path,
    out_dir: Path,
    crop_size: int = 0,
    slice_axis: int = 0,
):
    img_ras = as_ras(nib.load(str(vol_path)))
    lab_img_ras = as_ras(nib.load(str(label_path)))

    img = img_ras.get_fdata(dtype=np.float32)
    lab = lab_img_ras.get_fdata(dtype=np.float32)

    if img.ndim != 3 or lab.ndim != 3:
        raise ValueError("Expected 3D NIfTI")

    if img.shape != lab.shape:
        raise ValueError("Shape mismatch after RAS")

    mask = (lab > 0.5).astype(np.uint8)
    idx = pick_slice_with_positive(mask, axis=slice_axis)
    if idx < 0:
        return

    # Extract the 2D slice along the acquisition axis and orient Z(I→S) as height
    if slice_axis == 0:
        # Sagittal: slice is (Y=A→P, Z=I→S) → transpose to (Z, Y)
        img_2d = np.transpose(img[idx, :, :], (1, 0))
        mask_2d = np.transpose(mask[idx, :, :], (1, 0))
    elif slice_axis == 2:
        # Axial: slice is (X=R→L, Y=A→P) → transpose to (Y, X)
        img_2d = np.transpose(img[:, :, idx], (1, 0))
        mask_2d = np.transpose(mask[:, :, idx], (1, 0))
    else:
        raise ValueError(f"Unsupported slice_axis={slice_axis}")

    if crop_size > 0:
        img_2d = crop_around_positive(img_2d, mask_2d, size=crop_size)
        mask_2d = crop_around_positive(mask_2d, mask_2d, size=crop_size)

    js = load_json_sidecar(label_path)
    sev = js.get("PathologySeverity", None)

    if sev not in SEVERITY_TO_CLASS:
        raise ValueError(f"Unknown PathologySeverity '{sev}'")

    X = SEVERITY_TO_CLASS[sev]
    class_dir = out_dir / str(X)
    class_dir.mkdir(parents=True, exist_ok=True)

    out_name = label_path.name.replace(".nii.gz", ".npz")
    out_path = class_dir / out_name

    np.savez_compressed(out_path, slice=img_2d.astype(np.float32), mask=mask_2d.astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="/home/ge.polymtl.ca/p123239/data_ok/lumbar-rsna-challenge-2024")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["nfn", "ss", "scs"],
        help=(
            "nfn: T1w NeuralForaminalNarrowing | "
            "ss: T2w SubarticularStenosis | "
            "scs: T2w SpinalCanalStenosis"
        ),
    )
    ap.add_argument("--crop-size", type=int, default=0,
                    help="If > 0, crop a square patch of this size around the annotation. 0 = full slice.")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)

    from tqdm import tqdm

    pattern = TASK_CONFIG[args.task]["vol_pattern"]
    vol_paths = sorted(root.glob(pattern))

    for vol_path in tqdm(vol_paths, desc="Volumes", unit="vol"):
        labels = find_matching_labels(root, vol_path, args.task)

        for label_path in tqdm(labels, desc=f"Labels ({vol_path.name})", unit="lbl", leave=False):
            try:
                extract_and_save_patch(
                    vol_path,
                    label_path,
                    out_dir,
                    crop_size=args.crop_size,
                    slice_axis=TASK_CONFIG[args.task]["slice_axis"],
                )
            except Exception as e:
                print(f"[ERROR] {vol_path.name} + {label_path.name}: {e}")


if __name__ == "__main__":
    main()
