#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
        print(f"[cleanup] removed {path}")


def require_dir(path: Path, stage_name: str) -> None:
    if not path.exists() or not path.is_dir():
        raise RuntimeError(f"Cannot start from {stage_name}: missing folder {path}")


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
    ap.add_argument("--tile-overlap-pct", type=float, default=25.0)
    ap.add_argument("--tile-threshold", type=int, default=512)
    ap.add_argument("--target-spacing", type=float, default=0.8)
    ap.add_argument("--interp", type=str, default="bilinear", choices=["nearest", "bilinear", "bicubic", "lanczos"])
    ap.add_argument("--keep-intermediate", action="store_true", help="Keep intermediate folders for debugging")
    ap.add_argument("--start-stage", type=int, default=1, choices=[1, 2, 3, 4], help="Resume from stage: 1=extract, 2=resample, 3=tile+renumber, 4=sanity")
    ap.set_defaults(skip_existing=True)
    ap.add_argument("--skip-existing", dest="skip_existing", action="store_true", help="Skip files already processed in stages 1/2/3")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false", help="Recompute and overwrite checks in stages 1/2/3")
    args = ap.parse_args()

    print("=== Pipeline config ===")
    print(f"input-images     : {args.input_images}")
    print(f"input-labels     : {args.input_labels}")
    print(f"work-root        : {args.work_root}")
    print(f"train-ratio      : {args.train_ratio}")
    print(f"seed             : {args.seed}")
    print(f"tile-size        : {args.tile_size}")
    print(f"tile-overlap-pct : {args.tile_overlap_pct}")
    print(f"tile-threshold   : {args.tile_threshold}")
    print(f"target-spacing   : {args.target_spacing}")
    print(f"interp           : {args.interp}")
    print(f"keep-intermediate: {args.keep_intermediate}")
    print(f"start-stage      : {args.start_stage}")
    print(f"skip-existing    : {args.skip_existing}")
    print("=======================")

    this_dir = Path(__file__).resolve().parent
    extract_py = this_dir / "01_extract_slices.py"
    resample_py = this_dir / "02_resample_inplane.py"
    renumber_py = this_dir / "03_renumber_pairs.py"
    sanity_py = this_dir / "04_sanity_check_pairs.py"

    out_extract = args.work_root / "01_extracted"
    out_resample = args.work_root / "02_resampled"
    out_final = args.work_root / "03_final"
    t0 = time.perf_counter()

    if args.start_stage >= 2:
        require_dir(out_extract, "stage 2")
    if args.start_stage >= 3:
        require_dir(out_resample, "stage 3")
    if args.start_stage >= 4:
        require_dir(out_final, "stage 4")

    try:
        if args.start_stage <= 1:
            print("[stage 1/4] extraction")
            t_stage = time.perf_counter()
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
                ]
                + (["--skip-existing"] if args.skip_existing else ["--no-skip-existing"])
                + (["--iso-eps-mm", str(args.iso_eps_mm)] if args.iso_eps_mm is not None else [])
            )
            print(f"[stage 1/4 done] {time.perf_counter() - t_stage:.1f}s")
        else:
            print("[stage 1/4] skipped (resume)")

        if args.start_stage <= 2:
            print("[stage 2/4] in-plane resample")
            t_stage = time.perf_counter()
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
                + (["--skip-existing"] if args.skip_existing else ["--no-skip-existing"])
            )
            print(f"[stage 2/4 done] {time.perf_counter() - t_stage:.1f}s")
            if not args.keep_intermediate and args.start_stage <= 1:
                remove_tree(out_extract)
        else:
            print("[stage 2/4] skipped (resume)")

        if args.start_stage <= 3:
            print("[stage 3/4] renumber pairs")
            t_stage = time.perf_counter()
            run_cmd(
                [
                    sys.executable,
                    str(renumber_py),
                    "--src-root",
                    str(out_resample),
                    "--dst-root",
                    str(out_final),
                    "--tile-size",
                    str(args.tile_size),
                    "--tile-overlap-pct",
                    str(args.tile_overlap_pct),
                    "--tile-threshold",
                    str(args.tile_threshold),
                ]
                + (["--skip-existing"] if args.skip_existing else ["--no-skip-existing"])
            )
            print(f"[stage 3/4 done] {time.perf_counter() - t_stage:.1f}s")
            if not args.keep_intermediate and args.start_stage <= 2:
                remove_tree(out_resample)
        else:
            print("[stage 3/4] skipped (resume)")

        if args.start_stage <= 4:
            print("[stage 4/4] sanity check")
            t_stage = time.perf_counter()
            run_cmd(
                [
                    sys.executable,
                    str(sanity_py),
                    "--root",
                    str(out_final),
                ]
            )
            print(f"[stage 4/4 done] {time.perf_counter() - t_stage:.1f}s")
        else:
            print("[stage 4/4] skipped (resume)")
    finally:
        if not args.keep_intermediate:
            if args.start_stage <= 1:
                remove_tree(out_extract)
            if args.start_stage <= 2:
                remove_tree(out_resample)

    print("[done] Pipeline completed")
    print(f"[final] {out_final}")
    print(f"[total-time] {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
