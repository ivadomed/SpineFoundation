#!/usr/bin/env python
import argparse
import os
import glob
import sys

import numpy as np
import nibabel as nib
from nibabel.orientations import aff2axcodes
from tqdm import tqdm
import matplotlib.pyplot as plt


def find_nii_files(folder: str):
    pattern = "**/*.nii.gz"
    files = glob.glob(os.path.join(folder, pattern), recursive=True)
    files = [f for f in files
             if os.path.isfile(f)
             and "derivatives" not in f.lower()
             and "preproc" not in f.lower()]

    files = sorted(files)
    return files


def analyze_files(files):
    """
    Pour chaque NIfTI :
    - lit affine et orientation (axcodes) sans réorientation
    - assigne chaque dimension à X/Y/Z suivant:
        R/L -> X
        A/P -> Y
        S/I -> Z
    - calcule tailles physiques en mm pour X/Y/Z
    Retourne une liste de dicts.
    """
    records = []
    for f in tqdm(files, desc="Analyzing NIfTI files"):
        rec = {"filename": f}
        try:
            img = nib.load(f)
            hdr = img.header
            shape = img.shape
            zooms = hdr.get_zooms()

            if len(shape) < 3 or len(zooms) < 3:
                raise ValueError(f"Not enough spatial dims: shape={shape}, zooms={zooms}")

            axcodes = aff2axcodes(img.affine)  # ex: ('R','A','S')
            if axcodes is None or len(axcodes) < 3:
                raise ValueError(f"Cannot determine orientation codes: {axcodes}")

            # init
            sx = sy = sz = np.nan
            shx = shy = shz = np.nan

            # on ne regarde que les 3 premières dims spatiales
            for dim in range(3):
                code = axcodes[dim]  # 'R','L','A','P','S','I'
                dim_len = int(shape[dim])
                dim_zoom = float(zooms[dim])

                if code in ("R", "L"):
                    shx = dim_len
                    sx = dim_zoom
                elif code in ("A", "P"):
                    shy = dim_len
                    sy = dim_zoom
                elif code in ("S", "I"):
                    shz = dim_len
                    sz = dim_zoom
                else:
                    # cas très exotique
                    raise ValueError(f"Unknown orientation code {code} for dim {dim}")

            # vérif qu'on a bien trouvé les trois axes
            if any(np.isnan([sx, sy, sz, shx, shy, shz])):
                raise ValueError(
                    f"Incomplete RAS mapping: "
                    f"sx={sx}, sy={sy}, sz={sz}, shx={shx}, shy={shy}, shz={shz}, axcodes={axcodes}"
                )

            # tailles physiques en mm (dans le repère RAS logique)
            size_mm_x = shx * sx
            size_mm_y = shy * sy
            size_mm_z = shz * sz

            rec.update(
                spacing_x=sx,
                spacing_y=sy,
                spacing_z=sz,
                shape_x=shx,
                shape_y=shy,
                shape_z=shz,
                size_mm_x=size_mm_x,
                size_mm_y=size_mm_y,
                size_mm_z=size_mm_z,
                axcodes="".join(axcodes),
            )

        except Exception as e:
            rec.update(
                spacing_x=np.nan,
                spacing_y=np.nan,
                spacing_z=np.nan,
                shape_x=np.nan,
                shape_y=np.nan,
                shape_z=np.nan,
                size_mm_x=np.nan,
                size_mm_y=np.nan,
                size_mm_z=np.nan,
                error=str(e),
            )

        records.append(rec)

    return records


def plot_size_mm(records, out_png: str = "size_mm_hist.png"):
    """Plot des distributions de taille physique (mm) pour X/Y/Z et sauvegarde en PNG."""
    size_x = []
    size_y = []
    size_z = []
    for rec in records:
        if any(np.isnan([rec["size_mm_x"],
                         rec["size_mm_y"],
                         rec["size_mm_z"]])):
            continue
        size_x.append(rec["size_mm_x"])
        size_y.append(rec["size_mm_y"])
        size_z.append(rec["size_mm_z"])

    if not size_x:
        return

    size_x = np.array(size_x)
    size_y = np.array(size_y)
    size_z = np.array(size_z)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.hist(size_x, bins=40)
    plt.xlabel("Size X (mm) [R-L]")
    plt.ylabel("Count")
    plt.title("Physical size X")

    plt.subplot(1, 3, 2)
    plt.hist(size_y, bins=40)
    plt.xlabel("Size Y (mm) [A-P]")
    plt.title("Physical size Y")

    plt.subplot(1, 3, 3)
    plt.hist(size_z, bins=40)
    plt.xlabel("Size Z (mm) [S-I]")
    plt.title("Physical size Z")

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
