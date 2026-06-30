"""Remove all derived keys from NPZ files, keeping only base data (slice, mask, spacing_mm)."""

import os
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

KEEP_KEYS = {"slice", "mask", "spacing_mm"}

DATA_DIRS = [
    "/home/ge.polymtl.ca/p123239/data/RSNA_patches_scs",
    "/home/ge.polymtl.ca/p123239/data/RSNA_patches_nfn",
    "/home/ge.polymtl.ca/p123239/data/RSNA_patches_ss",
]


def _strip_one(path: str) -> tuple[str, bool, str]:
    """Keep only KEEP_KEYS in path. Returns (path, changed, error_msg)."""
    try:
        d = np.load(path, allow_pickle=False)
        existing = set(d.keys())
        to_remove = existing - KEEP_KEYS
        if not to_remove:
            return path, False, ""

        data = {k: d[k] for k in existing & KEEP_KEYS}
        dirpath = os.path.dirname(path)
        fd, tmp = tempfile.mkstemp(dir=dirpath, suffix=".npz")
        os.close(fd)
        try:
            np.savez_compressed(tmp, **data)
            os.replace(tmp, path)
        except Exception:
            os.unlink(tmp)
            raise
        return path, True, ""
    except Exception as e:
        return path, False, str(e)


def main():
    all_paths = []
    for d in DATA_DIRS:
        all_paths.extend(str(p) for p in Path(d).rglob("*.npz"))

    print(f"Found {len(all_paths)} NPZ files across {len(DATA_DIRS)} directories")
    print(f"Keeping: {sorted(KEEP_KEYS)}")
    print(f"Removing: patch_tokens* keys\n")

    n_workers = min(16, os.cpu_count() or 4)
    changed = errors = 0

    with Pool(n_workers) as pool:
        with tqdm(total=len(all_paths), unit="file") as pbar:
            for _, did_change, err in pool.imap_unordered(_strip_one, all_paths, chunksize=32):
                if did_change:
                    changed += 1
                if err:
                    errors += 1
                    tqdm.write(f"ERROR: {err}")
                pbar.update(1)
                pbar.set_postfix(changed=changed, errors=errors)

    print(f"\nDone. {changed} files rewritten, {errors} errors.")


if __name__ == "__main__":
    main()
