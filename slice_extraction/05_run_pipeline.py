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
from typing import Any


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML config file and return a flat dict of argparse-compatible keys.

    YAML keys use underscores (``work_root``); hyphens also accepted.
    Boolean YAML values are kept as-is; lists stay as lists.
    """
    try:
        import yaml  # pyyaml
    except ImportError:
        raise SystemExit(
            "pyyaml is required to read a config file: pip install pyyaml"
        )

    with open(config_path) as fh:
        raw = yaml.safe_load(fh) or {}

    # Normalise keys: replace hyphens with underscores so they match dest names.
    return {k.replace("-", "_"): v for k, v in raw.items()}


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


def rmtree_force(path: Path) -> None:
    """chmod -R u+w then rm -rf (needed because git-annex sets files read-only)."""
    run_cmd(["chmod", "-R", "u+w", str(path)])
    shutil.rmtree(path)
    print(f"[cleanup] removed {path}")


def stage_clone(repos: list[str], clone_root: Path) -> None:
    """Clone each repo only (git annex get is deferred to per-repo extraction)."""
    if not repos:
        print("[stage 1/5] clone: no --repos provided, skipping")
        return

    clone_root.mkdir(parents=True, exist_ok=True)
    for url in repos:
        name = repo_dir_name(url)
        dest = clone_root / name
        if dest.exists():
            print(f"[clone] already exists: {dest}, skipping git clone")
        else:
            run_cmd(["git", "clone", url, str(dest)])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="End-to-end slice-extraction pipeline (BIDS-friendly).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Config file (parsed first so CLI flags can override it) ───────────────
    ap.add_argument(
        "--config", type=Path, default=None, metavar="PATH",
        help="Path to a YAML config file. CLI flags override config values.",
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
    ap.add_argument("--git-annex-exclude", nargs="*", default=["derivatives/**"],
                    metavar="GLOB",
                    help="Glob patterns to exclude from 'git annex get' (default: derivatives/**).")

    # ── Stage 2: extract ──────────────────────────────────────────────────────
    ap.add_argument("--input-images", type=Path, nargs="*", default=None,
                    metavar="PATH",
                    help="One or more root directories of NIfTI images. "
                         "If omitted and --repos is given, defaults to --clone-root. "
                         "Can be combined with --repos to mix local folders and GitHub repos.")
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
    ap.add_argument("--workers", type=int, default=8,
                    help="Number of parallel worker processes for slice extraction (default: 8).")

    # ── Stage 3: renumber/tile ────────────────────────────────────────────────
    ap.set_defaults(renumber=True)
    ap.add_argument("--renumber", dest="renumber", action="store_true",
                    help="Run renumber+tile stage (default: on).")
    ap.add_argument("--no-renumber", dest="renumber", action="store_false",
                    help="Skip renumber stage entirely; 01_extracted is used directly as final output.")
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
    ap.add_argument("--work-root", type=Path, default=None,
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

    # First pass: extract --config only, then inject config values as defaults
    # so that any explicit CLI flag still takes precedence over the config file.
    pre, _ = ap.parse_known_args()
    if pre.config is not None:
        cfg_dict = _load_config(pre.config)
        # Path-type arguments must stay as strings here; argparse will convert.
        ap.set_defaults(**cfg_dict)

    args = ap.parse_args()

    if args.work_root is None:
        ap.error("--work-root is required (or set work_root in the config file).")

    # ── Resolve paths ─────────────────────────────────────────────────────────
    clone_root   = args.clone_root or (args.work_root / "00_cloned")
    input_images = list(args.input_images) if args.input_images else []
    if not input_images and not args.repos and args.start_stage <= 2:
        ap.error("--input-images or --repos is required.")

    with_labels = args.input_labels is not None

    out_extract  = args.work_root / "01_extracted"
    out_final    = out_extract if not args.renumber else args.work_root / "02_final"
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
    print(f"git-annex-exclude: {args.git_annex_exclude}")
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
    print(f"workers          : {args.workers}")
    print(f"skip-existing    : {args.skip_existing}")
    print("=======================")

    t0 = time.perf_counter()

    try:
        # ── Stage 1: skipped — clone+get+extract+rm is done per-repo in stage 2 ─
        if args.start_stage <= 1:
            print("[stage 1/5] clone skipped (handled per-repo in stage 2)")

        # ── Stage 2: extract slices ───────────────────────────────────────────
        if args.start_stage <= 2:
            n_sources = len(input_images) + len(args.repos)
            print(f"[stage 2/5] extraction ({n_sources} source(s))")
            t_stage = time.perf_counter()
            base_cmd = [
                sys.executable, str(extract_py),
                "--output-root",  str(out_extract),
                "--train-ratio",  str(args.train_ratio),
                "--seed",         str(args.seed),
                "--clip-pct",     str(args.clip_pct[0]), str(args.clip_pct[1]),
                "--iso-tol",      str(args.iso_tol),
                "--label-suffix", args.label_suffix,
            ]
            if with_labels:
                base_cmd += ["--input-labels", str(args.input_labels)]
            if args.iso_eps_mm is not None:
                base_cmd += ["--iso-eps-mm", str(args.iso_eps_mm)]
            base_cmd += ["--skip-existing"] if args.skip_existing else ["--no-skip-existing"]
            base_cmd += ["--workers", str(args.workers)]

            # Local folders first (no annex logic needed).
            for src in input_images:
                dataset_name = Path(src).name
                print(f"[stage 2/5] source (local): {src}  [dataset={dataset_name}]")
                run_cmd(base_cmd + ["--input-images", str(src), "--dataset-name", dataset_name])

            # Repos: clone → annex get → extract → rm -rf (one at a time to save disk).
            clone_root.mkdir(parents=True, exist_ok=True)
            for url in args.repos:
                name = repo_dir_name(url)
                dest = clone_root / name

                # Skip entirely if already extracted (any split directory contains this dataset).
                already_extracted = any(
                    (out_extract / "image" / split / name).exists()
                    for split in ("train", "val", "test")
                )
                if already_extracted:
                    print(f"[stage 2/5] [{name}] already extracted — skipping")
                    continue

                print(f"[stage 2/5] [{name}] git clone...")
                run_cmd(["git", "clone", url, str(dest)])

                if args.git_annex:
                    print(f"[stage 2/5] [{name}] git annex get .")
                    annex_cmd = ["git", "annex", "get"]
                    if args.git_annex_jobs > 1:
                        annex_cmd += [f"--jobs={args.git_annex_jobs}"]
                    for pat in (args.git_annex_exclude or []):
                        annex_cmd += [f"--exclude={pat}"]
                    annex_cmd.append(".")
                    print("[run]", " ".join(annex_cmd))
                    result = subprocess.run(annex_cmd, cwd=dest)
                    if result.returncode not in (0, 1):
                        raise subprocess.CalledProcessError(result.returncode, annex_cmd)
                    if result.returncode == 1:
                        print(f"[warn] git annex get finished with partial failures (exit 1) — continuing")

                print(f"[stage 2/5] [{name}] extracting slices...")
                run_cmd(base_cmd + ["--input-images", str(dest), "--dataset-name", name])

                print(f"[stage 2/5] [{name}] deleting repo...")
                rmtree_force(dest)

            print(f"[stage 2/5 done] {time.perf_counter() - t_stage:.1f}s")
        else:
            print("[stage 2/5] skipped (resume)")

        # ── Stage 3: renumber + tile ──────────────────────────────────────────
        if not args.renumber:
            print("[stage 3/5] renumber skipped (--no-renumber): using 01_extracted as final output")
        elif args.start_stage <= 3:
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
        if not args.keep_intermediate and args.renumber:
            if args.start_stage <= 2:
                remove_tree(out_extract)

    print("[done] Pipeline completed")
    final_out = out_resample if (args.resample and args.start_stage <= 5) else out_final
    print(f"[final] {final_out}")
    print(f"[total-time] {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
