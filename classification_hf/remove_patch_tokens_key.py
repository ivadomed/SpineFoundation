"""
Remove a specific key from all NPZ files in a dataset directory.
Usage:
    python remove_patch_tokens_key.py --data_dir /path/to/data --key patch_tokens_custom
"""
import argparse
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm


def remove_key_from_npz(path: str, key: str) -> str | None:
    """Remove key from NPZ file in-place. Returns path if modified, None if skipped."""
    try:
        d = np.load(path)
        if key not in d.files:
            return None
        existing = {k: d[k] for k in d.files if k != key}
        dirpath = os.path.dirname(path)
        fd, tmp_path = tempfile.mkstemp(dir=dirpath, suffix=".npz")
        os.close(fd)
        try:
            np.savez_compressed(tmp_path, **existing)
            os.replace(tmp_path, path)
        except Exception:
            os.unlink(tmp_path)
            raise
        return path
    except Exception as e:
        return f"ERROR:{path}:{e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    all_paths = [str(f) for f in data_path.rglob("*.npz")]
    print(f"Found {len(all_paths)} NPZ files in {data_path}")
    print(f"Removing key '{args.key}' ...")

    modified = 0
    errors = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {pool.submit(remove_key_from_npz, p, args.key): p for p in all_paths}
        for fut in tqdm(as_completed(futures), total=len(all_paths), unit="file"):
            result = fut.result()
            if result is None:
                pass  # key not present, skipped
            elif result.startswith("ERROR:"):
                errors.append(result)
            else:
                modified += 1

    print(f"\nDone. Modified: {modified} files.")
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  {e}")


if __name__ == "__main__":
    main()
