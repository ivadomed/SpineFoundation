#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-images", type=Path, required=True)
    ap.add_argument("--input-labels", type=Path, required=True)
    ap.add_argument("--work-root", type=Path, required=True, help="Root folder used for intermediate/final outputs")
    ap.add_argument("--train-ratio", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clip-pct", type=float, nargs=2, default=(0.5, 99.5))
    ap.add_argument("--iso-tol", type=float, default=0.1)
    ap.add_argument("--iso-eps-mm", type=float, default=None)
    ap.add_argument("--tile-size", type=int, default=224)
    ap.add_argument("--tile-overlap", type=int, default=56)
    ap.add_argument("--tile-threshold", type=int, default=512)
    ap.add_argument("--target-spacing", type=float, default=0.8)
    ap.add_argument("--interp", type=str, default="bilinear", choices=["nearest", "bilinear", "bicubic", "lanczos"])
    args = ap.parse_args()

    this_dir = Path(__file__).resolve().parent
    extract_py = this_dir / "01_extract_slices.py"
    resample_py = this_dir / "02_resample_inplane.py"
    renumber_py = this_dir / "03_renumber_pairs.py"
    sanity_py = this_dir / "04_sanity_check_pairs.py"

    out_extract = args.work_root / "01_extracted"
    out_resample = args.work_root / "02_resampled"
    out_final = args.work_root / "03_final"

    run_cmd(
        [
            sys.executable,
            str(extract_py),
            "--input-images",
            str(args.input_images),
            "--input-labels",
            str(args.input_labels),
            "--output-root",
            str(out_extract),
            "--train-ratio",
            str(args.train_ratio),
            "--seed",
            str(args.seed),
            "--clip-pct",
            str(args.clip_pct[0]),
            str(args.clip_pct[1]),
            "--iso-tol",
            str(args.iso_tol),
            "--tile-size",
            str(args.tile_size),
            "--tile-overlap",
            str(args.tile_overlap),
            "--tile-threshold",
            str(args.tile_threshold),
        ]
        + (["--iso-eps-mm", str(args.iso_eps_mm)] if args.iso_eps_mm is not None else [])
    )

    run_cmd(
        [
            sys.executable,
            str(resample_py),
            "--root",
            str(out_extract),
            "--target",
            str(args.target_spacing),
            "--interp",
            str(args.interp),
            "--out-root",
            str(out_resample),
        ]
    )

    run_cmd(
        [
            sys.executable,
            str(renumber_py),
            "--src-root",
            str(out_resample),
            "--dst-root",
            str(out_final),
        ]
    )

    run_cmd(
        [
            sys.executable,
            str(sanity_py),
            "--root",
            str(out_final),
        ]
    )

    print("[done] Pipeline completed")
    print(f"[final] {out_final}")


if __name__ == "__main__":
    main()
