"""
merge_embedding_cache.py

Appends embeddings from a secondary cache into the main cache.
Skips datasets that already exist in the main cache (idempotent).

Usage:
    python merge_embedding_cache.py \
        --main_cache   ./analysis_output \
        --extra_cache  ./analysis_output_ms_extra \
        --slug         models_curia_axial
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_cache",  required=True)
    parser.add_argument("--extra_cache", required=True)
    parser.add_argument("--slug",        default="models_curia_axial",
                        help="Filename slug, e.g. models_curia_axial")
    args = parser.parse_args()

    main_dir  = Path(args.main_cache)
    extra_dir = Path(args.extra_cache)
    slug      = args.slug

    emb_main   = main_dir  / f"embeddings_{slug}.npy"
    patch_main = main_dir  / f"patch_mean_{slug}.npy"
    meta_main  = main_dir  / f"metadata_{slug}.csv"

    emb_extra   = extra_dir / f"embeddings_{slug}.npy"
    patch_extra  = extra_dir / f"patch_mean_{slug}.npy"
    meta_extra   = extra_dir / f"metadata_{slug}.csv"

    for p in [emb_main, meta_main, emb_extra, meta_extra]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    print("Loading main cache…")
    emb_m   = np.load(emb_main)
    df_m    = pd.read_csv(meta_main)
    patch_m = np.load(patch_main) if patch_main.exists() else None

    print("Loading extra cache…")
    emb_e   = np.load(emb_extra)
    df_e    = pd.read_csv(meta_extra)
    patch_e = np.load(patch_extra) if patch_extra.exists() else None

    # Skip datasets already in main cache
    already = set(df_m["dataset"].unique())
    new_ds  = set(df_e["dataset"].unique()) - already
    if not new_ds:
        print("Nothing to merge — all datasets already in main cache.")
        return

    print(f"New datasets to add: {sorted(new_ds)}")
    keep = df_e["dataset"].isin(new_ds).values
    df_e    = df_e[keep].reset_index(drop=True)
    emb_e   = emb_e[keep]
    if patch_e is not None:
        patch_e = patch_e[keep]

    print(f"  Adding {len(df_e):,} rows to {len(df_m):,} existing rows")

    emb_merged   = np.concatenate([emb_m,   emb_e],   axis=0)
    df_merged    = pd.concat([df_m, df_e], ignore_index=True)

    # Backup originals
    emb_main.rename(emb_main.with_suffix(".npy.bak"))
    meta_main.rename(meta_main.with_suffix(".csv.bak"))

    np.save(emb_main, emb_merged)
    df_merged.to_csv(meta_main, index=False)
    print(f"  Saved merged embeddings: {emb_main}  shape={emb_merged.shape}")
    print(f"  Saved merged metadata  : {meta_main}  rows={len(df_merged):,}")

    if patch_m is not None and patch_e is not None:
        patch_main.rename(patch_main.with_suffix(".npy.bak"))
        patch_merged = np.concatenate([patch_m, patch_e], axis=0)
        np.save(patch_main, patch_merged)
        print(f"  Saved merged patch_mean: {patch_main}  shape={patch_merged.shape}")
    elif patch_m is None:
        print("  WARNING: no patch_mean in main cache — skipping patch merge")
    elif patch_e is None:
        print("  WARNING: no patch_mean in extra cache — skipping patch merge")

    print("Done.")


if __name__ == "__main__":
    main()
