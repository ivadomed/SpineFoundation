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
from pathlib import Path

def find_nii_files(folder: str):
    pattern = "**/*.nii.gz"
    files = glob.glob(os.path.join(folder, pattern), recursive=True)
    files = [f for f in files
             if os.path.isfile(f)
             and "derivatives" not in f.lower()
             and "preproc" not in f.lower()
             and ("lowres" not in f.lower() or Path(f.replace("lowres", "highres")).exists())]

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


def plot_spacings(records, out_png: str = "spacing_hist.png"):
    """Plot des distributions d'espacement (mm) pour X/Y/Z et sauvegarde en PNG."""
    spacing_x = []
    spacing_y = []
    spacing_z = []
    for rec in records:
        if any(np.isnan([rec["spacing_x"], rec["spacing_y"], rec["spacing_z"]])):
            continue
        spacing_x.append(rec["spacing_x"])
        spacing_y.append(rec["spacing_y"])
        spacing_z.append(rec["spacing_z"])

    if not spacing_x:
        return

    spacing_x = np.array(spacing_x)
    spacing_y = np.array(spacing_y)
    spacing_z = np.array(spacing_z)

    min_x = float(spacing_x.min())
    min_y = float(spacing_y.min())
    min_z = float(spacing_z.min())

    plt.figure(figsize=(12, 4))

    ax = plt.subplot(1, 3, 1)
    ax.hist(spacing_x, bins=40)
    ax.axvline(min_x, color='r', linestyle='--', linewidth=1)
    ax.set_xlabel("Spacing X (mm) [R-L]")
    ax.set_ylabel("Count")
    ax.set_title(f"Spacing X (min {min_x:.2f} mm)")

    ax = plt.subplot(1, 3, 2)
    ax.hist(spacing_y, bins=40)
    ax.axvline(min_y, color='r', linestyle='--', linewidth=1)
    ax.set_xlabel("Spacing Y (mm) [A-P]")
    ax.set_title(f"Spacing Y (min {min_y:.2f} mm)")

    ax = plt.subplot(1, 3, 3)
    ax.hist(spacing_z, bins=40)
    
    ax.axvline(min_z, color='r', linestyle='--', linewidth=1)
    ax.set_xlabel("Spacing Z (mm) [S-I]")
    ax.set_title(f"Spacing Z (min {min_z:.2f} mm)")

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def spacing_min_records(records):
    """Retourne les fichiers ayant les espacements minimums pour X/Y/Z."""
    minima = {}
    for axis in ("x", "y", "z"):
        key = f"spacing_{axis}"
        best = None
        for rec in records:
            val = rec.get(key, np.nan)
            if np.isnan(val):
                continue
            if best is None or val < best[0]:
                best = (val, rec.get("filename", ""))
        if best is not None:
            minima[axis] = best
    return minima


def parse_args():
    parser = argparse.ArgumentParser(description="Analyse des résolutions NIfTI et génération d'histogrammes")
    parser.add_argument("folder", help="Racine contenant les fichiers NIfTI à analyser")
    parser.add_argument("--size-plot", default="size_mm_hist.png", help="Nom du fichier PNG pour les tailles physiques")
    parser.add_argument("--spacing-plot", default="spacing_hist.png", help="Nom du fichier PNG pour les espacements")
    return parser.parse_args()


def main():
    args = parse_args()
    files = find_nii_files(args.folder)
    if not files:
        print(f"Aucun fichier NIfTI trouvé sous {args.folder}", file=sys.stderr)
        return 1

    print(f"Analyse de {len(files)} fichiers...")
    records = analyze_files(files)

    plot_size_mm(records, args.size_plot)
    plot_spacings(records, args.spacing_plot)

    mins = spacing_min_records(records)
    if mins:
        print("Espacements minimum observés :")
        for axis in ("x", "y", "z"):
            if axis in mins:
                val, fname = mins[axis]
                print(f"  axis {axis.upper()} : {val:.4f} mm -> {fname}")

    print(f"Histogrammes sauvegardés dans {args.size_plot} et {args.spacing_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    