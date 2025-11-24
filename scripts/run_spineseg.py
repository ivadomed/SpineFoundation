#!/usr/bin/env python3
"""
Run SCT `sct_deepseg totalspineseg` (step1-only) for images found under
provided data root folders.

For each image matching `*/sub-*/anat/*.nii.gz` the script will run:
    sct_deepseg totalspineseg -i <image> -o <outdir> -step1-only 1
where <outdir> = <root>/derivatives/labels/<sub-XXX>/anat

Usage:
    python scripts/run_spineseg.py [--dry-run] [--workers N] [root_folder ...]

If no root_folder is provided, defaults to `/home/ge.polymtl.ca/p123239/data`.
"""
import argparse
import concurrent.futures
import subprocess
import sys
import os
import re
from pathlib import Path
from shutil import which
from tqdm import tqdm


def find_images(root: Path):
    pattern = "**/sub-*/anat/*.nii.gz"
    return sorted(root.glob(pattern))


def run_for_image(img_path: Path, root: Path, dry_run: bool, step1_only: bool):
    # extract sub-XXX
    m = re.search(r"sub-[^/\\]+", str(img_path))
    if not m:
        return (str(img_path), False, "no-sub-found")
    sub = m.group(0)
    outdir = root / "derivatives" / "labels" / sub / "anat"
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = ["CUDA_VISIBLE_DEVICES=1", "SCT_USE_GPU=1", "sct_deepseg", "totalspineseg", "-i", str(img_path), "-o", str(outdir)]
    if step1_only:
        cmd += ["-step1-only", "1"]

    if dry_run:
        return (str(img_path), True, "dry-run: " + " ".join(cmd))

    try:
        res = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            return (str(img_path), False, f"returncode {res.returncode}: {res.stderr.strip()}")
        return (str(img_path), True, res.stdout.strip())
    except FileNotFoundError:
        return (str(img_path), False, "sct_deepseg not found in PATH")
    except Exception as e:
        return (str(img_path), False, str(e))


def main(argv=None):


    PATHS = ["/home/ge.polymtl.ca/p123239/data/lumbar-nusantara",
    "/home/ge.polymtl.ca/p123239/data/lumbar-rsna-challenge-2024",
    "/home/ge.polymtl.ca/p123239/data/ms-multi-spine-challenge-2024",
    "/home/ge.polymtl.ca/p123239/data/sci-zurich",
    "/home/ge.polymtl.ca/p123239/data/whole-spine"]
    workers = 1
    for p in PATHS:
        tasks = []
        root = Path(p)
        imgs = find_images(root)
        # filter views
        imgs = [p for p in imgs if ("ax" not in p.name.lower() and "cor" not in p.name.lower() and "preproc" not in p.name.lower())]
        for p in imgs:
            tasks.append((p, root))

        total = len(tasks)
        if total == 0:
            print("No images found. Exiting.")
            return 0

        print(f"Found {total} images across {len(PATHS)} root(s). Workers: {workers}. Dry-run: False")

        results = []
        if workers == 1:
            for img, root in tqdm(tasks, desc="Processing images"):
                res = run_for_image(img, root, dry_run=False, step1_only=True)
                results.append(res)

    # Summarize
    ok = [r for r in results if r[1] is True]
    fail = [r for r in results if r[1] is False]

    print(f"\nDone. Success: {len(ok)}, Failures: {len(fail)}")
    if len(fail) > 0:
        print("Failures (first 10):")
        for f in fail[:10]:
            print(f"  {f[0]} -> {f[2]}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
