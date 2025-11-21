
import argparse
import os
import glob
import sys
from collections import Counter
import nibabel as nib


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def find_nii_files(folder):
    pattern = "**/*.nii.gz"
    files = glob.glob(os.path.join(folder, pattern), recursive=True)
    all_files = sorted([f for f in files if "derivatives" not in f.lower() and "ax" not in f.lower() and "cor" not in f.lower() and "preproc" not in f.lower()])
    return all_files


def analyze_files(files):
    records = []
    for f in tqdm(files, desc="Scanning NIfTI files"):
        rec = {"filename": f}
        try:
            img = nib.load(f)
            hdr = img.header
            zo = hdr.get_zooms()
            # take first 3 zooms as spatial voxel sizes
            if len(zo) >= 3:
                # convert to numpy array for easy indexing
                zo_arr = np.array([float(zo[0]), float(zo[1]), float(zo[2])], dtype=float)
                # identify index of the largest spacing -> assign it to z
                z_idx = int(np.argmax(zo_arr))
                # keep the other two indices in their original order for x and y
                xy_indices = [i for i in range(3) if i != z_idx]
                # round spacings to 1e-4 precision
                sx = round(float(zo_arr[xy_indices[0]]), 4)
                sy = round(float(zo_arr[xy_indices[1]]), 4)
                sz = round(float(zo_arr[z_idx]), 4)
            else:
                # fallback if weird header
                zooms = list(zo) + [np.nan, np.nan, np.nan]
                sx, sy, sz = float(zooms[0]), float(zooms[1]), float(zooms[2])

            shape = img.shape
            # shape can be 3D or 4D; keep first 3 dims and reorder to match spacing assignment
            if len(shape) >= 3:
                sh = [int(shape[0]), int(shape[1]), int(shape[2])]
                if len(zo) >= 3:
                    shx = int(sh[xy_indices[0]])
                    shy = int(sh[xy_indices[1]])
                    shz = int(sh[z_idx])
                else:
                    shx, shy, shz = sh[0], sh[1], sh[2]
            elif len(shape) == 2:
                shx, shy, shz = int(shape[0]), int(shape[1]), 1
            else:
                shx = int(shape[0]) if len(shape) >= 1 else 1
                shy = 1
                shz = 1

            rec.update({
                "spacing_x": sx,
                "spacing_y": sy,
                "spacing_z": sz,
                "spacing": f"{sx:.4f}x{sy:.4f}x{sz:.4f}",
                "shape_x": shx,
                "shape_y": shy,
                "shape_z": shz,
                "shape": f"{shx}x{shy}x{shz}",
                "n_voxels": shx * shy * shz,
            })
        except Exception as e:
            rec.update({
                "spacing_x": np.nan,
                "spacing_y": np.nan,
                "spacing_z": np.nan,
                "spacing": None,
                "shape_x": None,
                "shape_y": None,
                "shape_z": None,
                "shape": None,
                "n_voxels": None,
                "error": str(e),
            })
        records.append(rec)
    return pd.DataFrame.from_records(records)


def summarize_df(df: pd.DataFrame):
    summary = {}
    total = len(df)
    summary["n_files"] = total
    valid = df[~df["spacing"].isnull()]
    summary["n_valid"] = len(valid)

    # unique spacing combos
    spacing_counts = valid["spacing"].value_counts()
    summary["most_common_spacings"] = spacing_counts.head(10).to_dict()

    # stats per axis
    for ax in ["x", "y", "z"]:
        col = f"spacing_{ax}"
        if col in df:
            arr = valid[col].dropna().values
            if arr.size:
                summary[f"spacing_{ax}_mean"] = round(float(np.mean(arr)), 4)
                summary[f"spacing_{ax}_median"] = round(float(np.median(arr)), 4)
                summary[f"spacing_{ax}_std"] = round(float(np.std(arr)), 4)
                summary[f"spacing_{ax}_min"] = round(float(np.min(arr)), 4)
                summary[f"spacing_{ax}_max"] = round(float(np.max(arr)), 4)
            else:
                summary[f"spacing_{ax}_mean"] = None

    # shapes
    # shape statistics (use rows where shape is not null)
    shape_valid = df[~df["shape"].isnull()]
    for ax in ["x", "y", "z"]:
        col = f"shape_{ax}"
        if col in df:
            arr = shape_valid[col].dropna().values
            if arr.size:
                summary[f"shape_{ax}_mean"] = round(float(np.mean(arr)), 4)
                summary[f"shape_{ax}_median"] = round(float(np.median(arr)), 4)
                summary[f"shape_{ax}_std"] = round(float(np.std(arr)), 4)
                summary[f"shape_{ax}_min"] = int(np.min(arr))
                summary[f"shape_{ax}_max"] = int(np.max(arr))
            else:
                summary[f"shape_{ax}_mean"] = None

    shape_counts = shape_valid["shape"].value_counts() if not shape_valid.empty else pd.Series(dtype=int)
    summary["most_common_shapes"] = shape_counts.head(10).to_dict()

    return summary


def plot_histograms(df: pd.DataFrame, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    valid = df[~df["spacing"].isnull()]
    for ax in ["x", "y", "z"]:
        col = f"spacing_{ax}"
        if col in df:
            arr = valid[col].dropna().values
            if arr.size:
                plt.figure()
                plt.hist(arr, bins=40)
                plt.xlabel(f"Spacing {ax} (mm)")
                plt.ylabel("Count")
                plt.title(f"Voxel spacing distribution ({ax})")
                plt.grid(True)
                out = os.path.join(outdir, f"hist_spacing_{ax}.png")
                plt.savefig(out, bbox_inches="tight")
                plt.close()

    # Histograms for shape (number of voxels per axis)
    for ax in ["x", "y", "z"]:
        col = f"shape_{ax}"
        if col in df:
            arr = df[col].dropna().values
            if arr.size:
                plt.figure()
                plt.hist(arr, bins=40)
                plt.xlabel(f"Shape {ax} (voxels)")
                plt.ylabel("Count")
                plt.title(f"Image shape distribution ({ax})")
                plt.grid(True)
                out = os.path.join(outdir, f"hist_shape_{ax}.png")
                plt.savefig(out, bbox_inches="tight")
                plt.close()

    # heatmap counts for spacing combos (x,y) as a simple scatter plot
    if not valid.empty:
        plt.figure(figsize=(6, 5))
        plt.scatter(valid["spacing_x"], valid["spacing_y"], s=20, alpha=0.6)
        plt.xlabel("spacing_x (mm)")
        plt.ylabel("spacing_y (mm)")
        plt.title("Spacing x vs y")
        out = os.path.join(outdir, "spacing_x_vs_y.png")
        plt.grid(True)
        plt.savefig(out, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze NIfTI resolutions in a folder")
    parser.add_argument("folder", help="Folder to scan for .nii or .nii.gz files")
    parser.add_argument("--plots", default=None, help="Output directory for plots (optional)")
    parser.add_argument("--min-files", type=int, default=0, help="Require at least N files to consider success")
    args = parser.parse_args()

    folder = args.folder
    if not os.path.isdir(folder):
        print(f"Error: {folder} is not a folder")
        sys.exit(2)

    files = find_nii_files(folder)
    print(f"Found {len(files)} NIfTI files in {folder}")

    df = analyze_files(files)
    # Compute and print statistics for spacing and shape
    summary = summarize_df(df)
    # Count non-square images (shape_x != shape_y)
    if "shape_x" in df and "shape_y" in df:
        shape_mask = (~df["shape_x"].isnull()) & (~df["shape_y"].isnull())
        n_with_shape = int(shape_mask.sum())
        n_non_square = int((df.loc[shape_mask, "shape_x"] != df.loc[shape_mask, "shape_y"]).sum())
        pct_non_square = 100.0 * n_non_square / max(1, n_with_shape)
        print(f"\nNon-square images: {n_non_square} / {n_with_shape} ({pct_non_square:.2f}%) where shape_x != shape_y")
    else:
        n_non_square = None
    print("\nSpacing statistics (mm):")
    for ax in ["x", "y", "z"]:
        mean_k = f"spacing_{ax}_mean"
        med_k = f"spacing_{ax}_median"
        std_k = f"spacing_{ax}_std"
        if mean_k in summary and summary.get(mean_k) is not None:
                print(f"  {ax}: mean={summary.get(mean_k):.4f}, median={summary.get(med_k):.4f}, std={summary.get(std_k):.4f}")

    print("\nShape statistics (voxels):")
    for ax in ["x", "y", "z"]:
        mean_k = f"shape_{ax}_mean"
        med_k = f"shape_{ax}_median"
        std_k = f"shape_{ax}_std"
        if mean_k in summary and summary.get(mean_k) is not None:
            print(f"  {ax}: mean={summary.get(mean_k):.4f}, median={summary.get(med_k):.4f}, std={summary.get(std_k):.4f}")

    if args.plots:
        plot_histograms(df, args.plots)
        print(f"Saved plots to: {args.plots}")



if __name__ == "__main__":
    main()
