#!/usr/bin/env python3
"""
sanity_check_imagenet_spine_filenames.py

Checks an ImageNet-like folder layout (class folder names ignored) AND a filename convention.

Folder layout (generic):
  <root>/{train,val,test}/<any_class_dir>/*.<img>

Filename convention (basename without extension):
  Must contain (in this order, with "__" separators for these tokens):
    ...__sub-<ID>...__<plane>__s<NNN...>__sp<XXXX>x<XXXX>__<UID>
  where:
    <plane> in {"sagittal","axial"}
    s-token is mandatory between plane and sp (e.g., s0009, s123, s01234)
    spacing token is mandatory: __sp0536x0536
    UID is an integer at the very end
    IMPORTANT: separator between spacing and UID is **FORCED** to "__" (not "_")

Example accepted:
  lumbar-rsna-challenge-2024__sub-1383495058_acq-sag_rec-..._T2w__sagittal__s0009__sp0536x0536__00705330.png

Usage:
  python sanity_check.py --root /home/ge.polymtl.ca/p123239/slice_extractor/out_raw_numero
    python sanity_check.py --root /home/ge.polymtl.ca/p123239/slice_extractor/out_raw_numero --check-dup-ids --dup-scope split
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# Require:
#  - somewhere: __sub-<something>
#  - then: __<plane>__s<digits>__sp<digits>x<digits>__<digits> END
RE_PATTERN = re.compile(
    r"""
    ^.*__sub-([A-Za-z0-9\-]+).*          # sub id somewhere after "__sub-"
    __(sagittal|axial)                   # plane token, preceded by "__"
    __s(\d{3,6})                         # mandatory slice token between plane and sp
    __sp(\d{3,4})x(\d{3,4})              # spacing token
    __(\d+)                              # FORCED separator "__" before unique id
    $                                    # end of basename
    """,
    re.IGNORECASE | re.VERBOSE,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Dataset root folder")
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"],
                    help="Splits to check (default: train val test)")
    ap.add_argument("--max-issues", type=int, default=200, help="Max issues to print")
    ap.add_argument("--allow-missing-splits", action="store_true",
                    help="Do not flag missing split folders as issues")
    ap.add_argument("--check-dup-ids", action="store_true",
                    help="Check duplicate UID (final numeric token) according to --dup-scope")
    ap.add_argument("--dup-scope", choices=["split", "global"], default="split",
                    help="Duplicate UID scope (default: split)")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[FATAL] Root does not exist: {root}")
        return 2

    issues = []
    stats = {
        "splits_found": 0,
        "class_dirs_found": 0,
        "images_found": 0,
        "non_image_files": 0,
        "bad_filenames": 0,
        "empty_class_dirs": 0,
    }

    def add_issue(msg: str):
        if len(issues) < args.max_issues:
            issues.append(msg)

    dup_maps = defaultdict(lambda: defaultdict(list))  # scope_key -> uid -> [paths]

    for split in args.splits:
        split_dir = root / split
        if not split_dir.exists():
            if not args.allow_missing_splits:
                add_issue(f"[SPLIT] Missing split folder: {split_dir}")
            continue
        if not split_dir.is_dir():
            add_issue(f"[SPLIT] Not a directory: {split_dir}")
            continue

        stats["splits_found"] += 1

        class_dirs = [p for p in split_dir.iterdir() if p.is_dir()]
        if not class_dirs:
            add_issue(f"[SPLIT] No class subfolders under: {split_dir}")

        for cdir in sorted(class_dirs):
            stats["class_dirs_found"] += 1
            files = list(cdir.iterdir())
            if not files:
                stats["empty_class_dirs"] += 1
                add_issue(f"[CLASS] Empty class folder: {cdir}")
                continue

            for fp in files:
                if fp.is_dir():
                    add_issue(f"[FILE] Unexpected subdirectory inside class folder: {fp}")
                    continue

                if fp.suffix.lower() not in IMG_EXTS:
                    stats["non_image_files"] += 1
                    add_issue(f"[FILE] Non-image file in class folder: {fp}")
                    continue

                stats["images_found"] += 1
                stem = fp.stem

                m = RE_PATTERN.match(stem)
                if m is None:
                    stats["bad_filenames"] += 1
                    add_issue(
                        f"[NAME] Bad filename: {fp.name} | expected tokens: "
                        f"...__sub-<ID>...__(sagittal|axial)__s<NNN...>__spXXXXxXXXX__<INT>.<ext>"
                    )
                    continue

                uid = m.group(6)
                scope_key = "global" if args.dup_scope == "global" else split
                dup_maps[scope_key][uid].append(fp)

    if args.check_dup_ids:
        for scope_key, idmap in dup_maps.items():
            for uid, paths in idmap.items():
                if len(paths) > 1:
                    add_issue(
                        f"[DUP:{scope_key}] Duplicate UID {uid}: "
                        + ", ".join(str(p.relative_to(root)) for p in paths)
                    )

    print("=== SANITY CHECK SUMMARY ===")
    print(f"Root: {root}")
    print(f"Splits checked: {', '.join(args.splits)}")
    print(f"Splits found: {stats['splits_found']}")
    print(f"Class dirs found: {stats['class_dirs_found']}")
    print(f"Images found: {stats['images_found']}")
    print(f"Empty class dirs: {stats['empty_class_dirs']}")
    print(f"Non-image files: {stats['non_image_files']}")
    print(f"Bad filenames: {stats['bad_filenames']}")
    if args.check_dup_ids:
        print(f"Duplicate UID check: ON (scope={args.dup_scope})")
    print()

    if issues:
        print(f"=== ISSUES (showing up to {args.max_issues}) ===")
        for msg in issues:
            print(msg)
        print()
        print(f"TOTAL ISSUES: {len(issues)} (printed up to {args.max_issues})")
        return 2

    print("No issues found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
