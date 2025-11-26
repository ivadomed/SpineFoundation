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
    pattern = "sub-*/**/anat/*.nii.gz"
    return sorted(root.glob(pattern))


def run_for_image(img_path: Path, root: Path, dry_run: bool, step1_only: bool):

    img=str(img_path)
    temp="/".join(img.split("/")[:-1])
    name ="".join(img.split("/")[-1])
    m = re.search(r"sub-[^/\\]+", str(temp))
    print(m)
    if not m:
        return (str(img_path), False, "no-sub-found")
    sub = m.group(0)
    outdir = root / "derivatives" / "labels" / sub / "anat" / "TTS" / name

    cmd = [
        "sct_deepseg",
        "totalspineseg",
        "-i", str(img_path),
        "-o", str(outdir),
        "-step1-only", "1",
    ]

    # On part de l'environnement actuel et on ajoute les variables GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    env["SCT_USE_GPU"] = "1"

    print("Running:", " ".join(cmd))
    if dry_run:
        return (str(img_path), True, "dry-run")

    # stdout/stderr non redirigés → tout s'affiche en direct dans le terminal
    res = subprocess.run(cmd, check=False, env=env)

    ok = (res.returncode == 0)
    return (str(img_path), ok, f"returncode={res.returncode}")




def main(argv=None):


    PATHS = ["/home/ge.polymtl.ca/p123239/data/lumbar-nusantara",
    "/home/ge.polymtl.ca/p123239/data/lumbar-rsna-challenge-2024",
    "/home/ge.polymtl.ca/p123239/data/ms-multi-spine-challenge-2024",
    "/home/ge.polymtl.ca/p123239/data/sci-zurich",
    "/home/ge.polymtl.ca/p123239/data/whole-spine"]

    total_img=[]
    for p in PATHS:
        tasks = []
        root = Path(p)
        imgs = find_images(root)
        # filter views
        imgs = [k for k in imgs if ("ax" not in k.name.lower() and "cor" not in k.name.lower() and "preproc" not in k.name.lower())]
        for i in imgs:
            tasks.append((i, root))

        total = len(tasks)
        total_img.extend(tasks)
        print(f"Found {total} images across {p}")

    results = []
    input(f"Cela va prendre environ {round(len(total_img)*100/60/60/24,2)} jours, entrée pour continuer CTRL C sinon.")

    for img, root in tqdm(total_img, desc="Processing images"):
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



if __name__ == '__main__':
    raise SystemExit(main())
