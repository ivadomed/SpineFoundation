#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import nibabel as nib


SEVERITY_TO_CLASS = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
LABEL_SUFFIX = "_label.nii.gz"


def load_json_sidecar(label_nii: Path) -> dict:
    p = Path(str(label_nii).replace("_label.nii.gz", "_label.json"))
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"No JSON sidecar found for label: {label_nii}")


def as_ras(img: nib.Nifti1Image) -> nib.Nifti1Image:
    return nib.as_closest_canonical(img)



def pick_sagittal_slice_with_positive(mask3d: np.ndarray) -> int:
    counts = mask3d.reshape(mask3d.shape[0], -1).sum(axis=1)
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


def find_matching_labels(root: Path, t2_path: Path):
    base = t2_path.name
    if not base.endswith("_T2w.nii.gz"):
        return []

    sub_id = t2_path.parent.parent.name
    labels_dir = root / "derivatives" / "labels" / sub_id / "anat"
    if not labels_dir.exists():
        return []

    prefix = base.replace("_T2w.nii.gz", "_T2w")

    labels = sorted(
        labels_dir.glob(prefix + "_desc-*_label-SpinalCanalStenosis_label.nii.gz")
    )
    return labels


def extract_and_save_patch(
    t1_path: Path,
    label_path: Path,
    out_dir: Path,
    crop_size: int = 0,
):
    t1_img_ras = as_ras(nib.load(str(t1_path)))
    lab_img_ras = as_ras(nib.load(str(label_path)))

    t1 = t1_img_ras.get_fdata(dtype=np.float32)
    lab = lab_img_ras.get_fdata(dtype=np.float32)

    if t1.ndim != 3 or lab.ndim != 3:
        raise ValueError("Expected 3D NIfTI")

    if t1.shape != lab.shape:
        raise ValueError("Shape mismatch after RAS")

    mask = (lab > 0.5).astype(np.uint8)
    x_idx = pick_sagittal_slice_with_positive(mask)
    if x_idx < 0:
        return

    # Transpose both image and mask: (Y,Z) → (Z,Y): Z (I→S) en hauteur, Y (P→A) en largeur
    t1_yz = np.transpose(t1[x_idx, :, :], (1, 0))
    mask_yz = np.transpose(mask[x_idx, :, :], (1, 0))

    if crop_size > 0:
        t1_yz = crop_around_positive(t1_yz, mask_yz, size=crop_size)

    js = load_json_sidecar(label_path)
    sev = js.get("PathologySeverity", None)

    if sev not in SEVERITY_TO_CLASS:
        raise ValueError(f"Unknown PathologySeverity '{sev}'")

    X = SEVERITY_TO_CLASS[sev]
    class_dir = out_dir / str(X)
    class_dir.mkdir(parents=True, exist_ok=True)

    out_name = label_path.name.replace(".nii.gz", ".npz")
    out_path = class_dir / out_name

    np.savez_compressed(out_path, slice=t1_yz.astype(np.float32), mask=mask_yz.astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--crop-size", type=int, default=0,
                    help="If > 0, crop a square patch of this size around the annotation. 0 = full slice.")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)

    from tqdm import tqdm

    pattern = "sub-*/anat/*_acq-sag_rec-*_T2w.nii.gz"
    t2_paths = sorted(root.glob(pattern))

    for t2_path in tqdm(t2_paths, desc="T2 volumes", unit="vol"):
        labels = find_matching_labels(root, t2_path)

        for label_path in tqdm(labels, desc=f"Labels ({t2_path.name})", unit="lbl", leave=False):
            try:
                extract_and_save_patch(
                    t2_path,
                    label_path,
                    out_dir,
                    crop_size=args.crop_size,
                )
            except Exception as e:
                print(f"[ERROR] {t2_path.name} + {label_path.name}: {e}")


if __name__ == "__main__":
    main()
