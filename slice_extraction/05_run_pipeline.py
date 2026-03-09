#!/usr/bin/env python3
"""End-to-end slice-extraction pipeline.

Stages
------
  1  clone   — git clone <repos> + git annex get .   (skipped if --repos empty)
  2  extract — 01_extract_slices.py
  3  rename  — 03_renumber_pairs.py  (tiling + renumber, applied on raw extract output)
  4  sanity  — 04_sanity_check_pairs.py
  5  resample— 02_resample_inplane.py  (optional, applied in-place on final output)

Use --start-stage N to resume from a specific stage.
Use --resample / --no-resample to enable/disable the optional in-plane resample (stage 5).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
        print(f"[cleanup] removed {path}")


def require_dir(path: Path, stage_name: str) -> None:
    if not path.exists() or not path.is_dir():
        raise RuntimeError(f"Cannot start from {stage_name}: missing folder {path}")


def repo_dir_name(url: str) -> str:
    """Derive a local folder name from a repo URL."""
    name = url.rstrip("/").split("/")[-1]
    if name.endswith(".git"):
        name = name[:-4]
    return name or "repo"


def stage_clone(repos: list[str], clone_root: Path, git_annex: bool, git_annex_jobs: int) -> None:
    """Clone each repo and optionally run git annex get ."""
    if not repos:
        print("[stage 1/5] clone: no --repos provided, skipping")
        return

    clone_root.mkdir(parents=True, exist_ok=True)
    for url in repos:
        name    = repo_dir_name(url)
        dest    = clone_root / name
        if dest.exists():
            print(f"[clone] already exists: {dest}, skipping git clone")
        else:
            run_cmd(["git", "clone", url, str(dest)])

        if git_annex:
            annex_cmd = ["git", "annex", "get"]
            if git_annex_jobs > 1:
                annex_cmd += [f"--jobs={git_annex_jobs}"]
            annex_cmd.append(".")
            run_cmd(annex_cmd, cwd=dest)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="End-to-end slice-extraction pipeline (BIDS-friendly).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Stage 1: clone ────────────────────────────────────────────────────────
    ap.add_argument("--repos", nargs="*", default=[],
                    metavar="URL",
                    help="One or more git repository URLs to clone before extracting. "
                         "Repos are cloned into --clone-root.")
    ap.add_argument("--clone-root", type=Path, default=None,
                    help="Directory where repos are cloned "
                         "(default: <work-root>/00_cloned).")
    ap.set_defaults(git_annex=True)
    ap.add_argument("--git-annex", dest="git_annex", action="store_true",
                    help="Run 'git annex get .' after cloning (default: on).")
    ap.add_argument("--no-git-annex", dest="git_annex", action="store_false",
                    help="Skip git annex get.")
    ap.add_argument("--git-annex-jobs", type=int, default=4,
                    help="Parallelism passed to 'git annex get --jobs'.")

    # ── Stage 2: extract ──────────────────────────────────────────────────────
    ap.add_argument("--input-images", type=Path, default=None,
                    help="Root directory of NIfTI images. "
                         "If omitted and --repos is given, defaults to --clone-root.")
    ap.add_argument("--input-labels", type=Path, default=None,
                    help="Root of label/derivative NIfTI files (BIDS derivatives).")
    ap.add_argument("--label-suffix", type=str, default="_seg",
                    help="Suffix appended to image stem to locate the label file "
                         "(e.g. '_seg' matches sub-01_T2w_seg.nii.gz).")
    ap.add_argument("--train-ratio", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clip-pct", type=float, nargs=2, default=(0.5, 99.5))
    ap.add_argument("--iso-tol", type=float, default=0.1)
    ap.add_argument("--iso-eps-mm", type=float, default=None)

    # ── Stage 3: renumber/tile ────────────────────────────────────────────────
    ap.set_defaults(tiling=True)
    ap.add_argument("--tiling", dest="tiling", action="store_true",
                    help="Enable tiling during renumber stage (default: on).")
    ap.add_argument("--no-tiling", dest="tiling", action="store_false",
                    help="Disable tiling.")
    ap.add_argument("--tile-size", type=int, default=224)
    ap.add_argument("--tile-overlap-pct", type=float, default=25.0)
    ap.add_argument("--tile-threshold", type=int, default=512)

    # ── Stage 5 (optional): resample ──────────────────────────────────────────
    ap.set_defaults(resample=False)
    ap.add_argument("--resample", dest="resample", action="store_true",
                    help="Enable optional in-plane resample as the last stage.")
    ap.add_argument("--no-resample", dest="resample", action="store_false",
                    help="Skip in-plane resample (default).")
    ap.add_argument("--target-spacing", type=float, default=0.8,
                    help="Target isotropic in-plane spacing (mm) for stage 5.")
    ap.add_argument("--interp", type=str, default="bilinear",
                    choices=["nearest", "bilinear", "bicubic", "lanczos"])

    # ── Common ────────────────────────────────────────────────────────────────
    ap.add_argument("--work-root", type=Path, required=True,
                    help="Root folder for intermediate and final outputs.")
    ap.add_argument("--keep-intermediate", action="store_true",
                    help="Keep intermediate folders for debugging.")
    ap.add_argument("--start-stage", type=int, default=1,
                    choices=[1, 2, 3, 4, 5],
                    help="Resume from stage: "
                         "1=clone, 2=extract, 3=renumber, 4=sanity, 5=resample.")
    ap.set_defaults(skip_existing=True)
    ap.add_argument("--skip-existing", dest="skip_existing", action="store_true",
                    help="Skip files already processed in stages 2/3.")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                    help="Recompute and overwrite in stages 2/3.")

    args = ap.parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────────────
    clone_root  = args.clone_root or (args.work_root / "00_cloned")
    input_images = args.input_images
    if input_images is None and args.repos:
        input_images = clone_root
        print(f"[info] --input-images not set; will use clone root: {input_images}")
    if input_images is None and args.start_stage <= 2:
        ap.error("--input-images is required when --repos is empty.")

    with_labels = args.input_labels is not None

    out_extract  = args.work_root / "01_extracted"
    out_final    = args.work_root / "02_final"
    out_resample = args.work_root / "03_resampled"

    this_dir     = Path(__file__).resolve().parent
    extract_py   = this_dir / "01_extract_slices.py"
    resample_py  = this_dir / "02_resample_inplane.py"
    renumber_py  = this_dir / "03_renumber_pairs.py"
    sanity_py    = this_dir / "04_sanity_check_pairs.py"

    # ── Require intermediate dirs when resuming ───────────────────────────────
    if args.start_stage >= 3:
        require_dir(out_extract, "stage 3 (renumber)")
    if args.start_stage >= 4:
        require_dir(out_final, "stage 4 (sanity)")
    if args.start_stage >= 5:
        require_dir(out_final, "stage 5 (resample)")

    print("=== Pipeline config ===")
    print(f"repos            : {args.repos}")
    print(f"clone-root       : {clone_root}")
    print(f"git-annex        : {args.git_annex}")
    print(f"git-annex-jobs   : {args.git_annex_jobs}")
    print(f"input-images     : {input_images}")
    print(f"input-labels     : {args.input_labels}")
    print(f"label-suffix     : {args.label_suffix}")
    print(f"with-labels      : {with_labels}")
    print(f"work-root        : {args.work_root}")
    print(f"train-ratio      : {args.train_ratio}")
    print(f"seed             : {args.seed}")
    print(f"tiling           : {args.tiling}")
    print(f"tile-size        : {args.tile_size}")
    print(f"tile-overlap-pct : {args.tile_overlap_pct}")
    print(f"tile-threshold   : {args.tile_threshold}")
    print(f"resample (opt.)  : {args.resample}")
    print(f"target-spacing   : {args.target_spacing}")
    print(f"interp           : {args.interp}")
    print(f"keep-intermediate: {args.keep_intermediate}")
    print(f"start-stage      : {args.start_stage}")
    print(f"skip-existing    : {args.skip_existing}")
    print("=======================")

    t0 = time.perf_counter()

    try:
        # ── Stage 1: clone repos + git annex get ─────────────────────────────
        if args.start_stage <= 1:
            print("[stage 1/5] clone + git annex get")
            t_stage = time.perf_counter()
            stage_clone(
                repos=args.repos,
                clone_root=clone_root,
                git_annex=args.git_annex,
                git_annex_jobs=args.git_annex_jobs,
            )
            print(f"[stage 1/5 done] {time.perf_counter() - t_stage:.1f}s")
        else:
            print("[stage 1/5] skipped (resume)")

        # ── Stage 2: extract slices ───────────────────────────────────────────
        if args.start_stage <= 2:
            print("[stage 2/5] extraction")
            t_stage = time.perf_counter()
            run_cmd(
                [
                    sys.executable, str(extract_py),
                    "--input-images", str(input_images),
                    "--output-root",  str(out_extract),
                    "--train-ratio",  str(args.train_ratio),
                    "--seed",         str(args.seed),
                    "--clip-pct",     str(args.clip_pct[0]), str(args.clip_pct[1]),
                    "--iso-tol",      str(args.iso_tol),
                    "--label-suffix", args.label_suffix,
                ]
                + (["--input-labels",  str(args.input_labels)] if with_labels else [])
                + (["--iso-eps-mm",    str(args.iso_eps_mm)]   if args.iso_eps_mm is not None else [])
                + (["--skip-existing"] if args.skip_existing else ["--no-skip-existing"])
            )
            print(f"[stage 2/5 done] {time.perf_counter() - t_stage:.1f}s")
        else:
            print("[stage 2/5] skipped (resume)")

        # ── Stage 3: renumber + tile ──────────────────────────────────────────
        if args.start_stage <= 3:
            print("[stage 3/5] renumber pairs")
            t_stage = time.perf_counter()
            run_cmd(
                [
                    sys.executable, str(renumber_py),
                    "--src-root", str(out_extract),
                    "--dst-root", str(out_final),
                    "--with-labels" if with_labels else "--no-labels",
                    "--tiling"    if args.tiling   else "--no-tiling",
                    "--tile-size",         str(args.tile_size),
                    "--tile-overlap-pct",  str(args.tile_overlap_pct),
                    "--tile-threshold",    str(args.tile_threshold),
                ]
                + (["--skip-existing"] if args.skip_existing else ["--no-skip-existing"])
            )
            print(f"[stage 3/5 done] {time.perf_counter() - t_stage:.1f}s")
            if not args.keep_intermediate and args.start_stage <= 2:
                remove_tree(out_extract)
        else:
            print("[stage 3/5] skipped (resume)")

        # ── Stage 4: sanity check ─────────────────────────────────────────────
        if args.start_stage <= 4:
            print("[stage 4/5] sanity check")
            t_stage = time.perf_counter()
            run_cmd(
                [
                    sys.executable, str(sanity_py),
                    "--root", str(out_final),
                    "--with-labels" if with_labels else "--no-labels",
                ]
            )
            print(f"[stage 4/5 done] {time.perf_counter() - t_stage:.1f}s")
        else:
            print("[stage 4/5] skipped (resume)")

        # ── Stage 5 (optional): in-plane resample ────────────────────────────
        if args.resample:
            if args.start_stage <= 5:
                print("[stage 5/5] in-plane resample (optional)")
                t_stage = time.perf_counter()
                run_cmd(
                    [
                        sys.executable, str(resample_py),
                        "--root",    str(out_final),
                        "--target",  str(args.target_spacing),
                        "--interp",  args.interp,
                        "--out-root", str(out_resample),
                        "--with-labels" if with_labels else "--no-labels",
                    ]
                    + (["--skip-existing"] if args.skip_existing else ["--no-skip-existing"])
                )
                print(f"[stage 5/5 done] {time.perf_counter() - t_stage:.1f}s")
            else:
                print("[stage 5/5] skipped (resume)")
        else:
            print("[stage 5/5] in-plane resample disabled (use --resample to enable)")

    finally:
        if not args.keep_intermediate:
            if args.start_stage <= 2:
                remove_tree(out_extract)

    print("[done] Pipeline completed")
    final_out = out_resample if (args.resample and args.start_stage <= 5) else out_final
    print(f"[final] {final_out}")
    print(f"[total-time] {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
