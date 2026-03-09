"""
Run a Dinov2 backbone over a flat dataset directory and save the raw (non-pooled)
patch tokens back into each NPZ file under a suffixed key.

NPZ files must contain a "slice" key (H, W float32 array).
After running this script, cached tokens can be used during training
to bypass the frozen backbone entirely (fast path).

    python -m segmentation_hf.cache_patch_tokens \\
        --data_dir /path/to/seg_slices_npz \\
        --model_name /path/to/backbone \\
        --processor_name /path/to/curia_snapshot \\
        --suffix custom

The NPZ files are updated in-place: all existing keys are preserved and
patch_tokens_{suffix} is added (or overwritten if --overwrite).

Optimisations vs version naïve :
  - Pré-check parallèle (16 threads) au lieu de séquentiel
  - Lecture NPZ unique par fichier (pas de double chargement)
  - Écriture savez_compressed asynchrone (8 threads) pendant que le GPU
    traite le batch suivant → GPU idle ≈ 0%
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from tqdm import tqdm

_NPZ_EXT = ".npz"


def _has_key(path: str, key: str) -> bool:
    """Return True if the NPZ already contains 'key' (reads only the header)."""
    try:
        return key in np.load(path).files
    except Exception:
        return False


def _load_npz(path: str, skip_key: str | None = None) -> tuple[str, dict, np.ndarray]:
    """Load NPZ, return (path, existing_dict, slice_array).

    If skip_key is set, that key is excluded from existing_dict (avoids
    loading large arrays we are about to overwrite anyway).
    """
    d = np.load(path)
    existing = {k: d[k] for k in d.files if k != skip_key}
    img = existing["slice"].astype(np.float32)
    return path, existing, img


def _save_npz(path: str, existing: dict, token_key: str, tokens_np: np.ndarray) -> None:
    """Write updated NPZ atomically via a temp file + os.replace().

    If the process is killed mid-write the original file is untouched,
    so restarting the script is always safe (idempotent).
    """
    existing[token_key] = tokens_np
    dirpath = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=dirpath, suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp_path, **existing)
        os.replace(tmp_path, path)   # atomic rename on POSIX
    except Exception:
        os.unlink(tmp_path)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",       required=True,
                        help="Flat directory containing NPZ files with a 'slice' key.")
    parser.add_argument("--model_name",     required=True,
                        help="Path to a Dinov2 backbone checkpoint.")
    parser.add_argument("--processor_name", default=None,
                        help="Path to AutoImageProcessor (defaults to --model_name). "
                             "Useful when the checkpoint has no preprocessor_config.json.")
    parser.add_argument("--suffix",         default=None,
                        help="Key suffix: tokens saved as patch_tokens_{suffix}. "
                             "If omitted, saved as plain 'patch_tokens'.")
    parser.add_argument("--batch_size",     type=int, default=64)
    parser.add_argument("--num_readers",    type=int, default=16,
                        help="Threads for parallel pre-check and NPZ reads (default: 16)")
    parser.add_argument("--num_writers",    type=int, default=8,
                        help="Threads for async NPZ writes (default: 8)")
    parser.add_argument("--overwrite",      action="store_true",
                        help="Re-compute and overwrite existing patch_tokens_{suffix} keys.")
    args = parser.parse_args()

    data_path      = Path(args.data_dir)
    token_key      = f"patch_tokens_{args.suffix}" if args.suffix else "patch_tokens"
    processor_name = args.processor_name or args.model_name

    # ── Collect all NPZ paths (flat directory, no class subdirs) ─────────────
    all_paths = sorted([str(f) for f in data_path.iterdir()
                        if f.is_file() and f.suffix.lower() == _NPZ_EXT])
    n = len(all_paths)
    if n == 0:
        raise ValueError(f"No NPZ files found in {data_path}")

    print(f"Found {n} NPZ files in {data_path}")

    # ── Parallel pre-check (reads only NPZ header, not the arrays) ────────────
    if not args.overwrite:
        print(f"Checking existing '{token_key}' keys ({args.num_readers} threads)...",
              flush=True)
        pending = []
        with ThreadPoolExecutor(max_workers=args.num_readers) as ex:
            futures = {ex.submit(_has_key, p, token_key): p for p in all_paths}
            for fut in tqdm(as_completed(futures), total=n,
                            desc="Pre-check", unit="file", leave=False):
                if not fut.result():
                    pending.append(futures[fut])
        skipped = n - len(pending)
        if skipped:
            print(f"Skipping {skipped} files already containing '{token_key}' "
                  f"(use --overwrite to force)")
        all_paths = pending

    if not all_paths:
        print("Nothing to do.")
        return

    print(f"To process       : {len(all_paths)} files")
    print(f"Backbone         : {args.model_name}")
    print(f"Processor        : {processor_name}")
    print(f"Token key        : {token_key}")
    print(f"Batch size       : {args.batch_size}")
    print(f"Writer threads   : {args.num_writers}")

    # ── Load backbone ─────────────────────────────────────────────────────────
    from transformers import AutoImageProcessor, Dinov2Model

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device           : {device}\n")

    processor = AutoImageProcessor.from_pretrained(processor_name, trust_remote_code=True)
    backbone  = Dinov2Model.from_pretrained(args.model_name, trust_remote_code=True)
    backbone.to(device)
    backbone.eval()

    # ── Process in batches ────────────────────────────────────────────────────
    # Pipeline: GPU(N) runs while writes(N-1) complete.
    t0 = time.time()
    errors = []
    skip_key = token_key if args.overwrite else None
    prev_futures: list = []

    def _wait_for(futures: list) -> None:
        for f in futures:
            try:
                f.result()
            except Exception as e:
                errors.append(str(e))

    with ThreadPoolExecutor(max_workers=args.num_readers) as reader_pool, \
         ThreadPoolExecutor(max_workers=args.num_writers) as writer_pool:
        with tqdm(total=len(all_paths), desc="Caching patch_tokens", unit="ex") as pbar:
            for start in range(0, len(all_paths), args.batch_size):
                batch_paths = all_paths[start : start + args.batch_size]

                # Parallel reads — overlap with previous batch's writes
                loaded = list(reader_pool.map(
                    lambda p: _load_npz(p, skip_key), batch_paths
                ))
                batch_existing = [x[1] for x in loaded]
                images_np      = [x[2] for x in loaded]

                # GPU inference
                processed    = processor(images_np, return_tensors="pt")
                pixel_values = processed["pixel_values"].to(device)
                with torch.no_grad():
                    outputs = backbone(pixel_values=pixel_values,
                                       output_hidden_states=False)

                # Skip CLS token → (B, N, D)
                patch_tokens_batch = outputs.last_hidden_state[:, 1:, :].cpu().float()

                # Wait for previous batch's writes (had the GPU window to complete)
                _wait_for(prev_futures)

                # Submit current batch's writes asynchronously
                prev_futures = [
                    writer_pool.submit(
                        _save_npz, path, batch_existing[i], token_key,
                        patch_tokens_batch[i].numpy()
                    )
                    for i, path in enumerate(batch_paths)
                ]

                pbar.update(len(batch_paths))

        # Wait for the last batch's writes
        if prev_futures:
            print(f"Waiting for {len(prev_futures)} final write(s)...", flush=True)
            _wait_for(prev_futures)

    if errors:
        print(f"\n[WARN] {len(errors)} write errors:")
        for e in errors[:5]:
            print(f"  {e}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s — '{token_key}' added to {len(all_paths)} NPZ files.")
    print(f"\nNext step: launch training with cached tokens:")
    print(f"  python -m segmentation_hf.train_seg_from_hf \\")
    print(f"      --model_dir <backbone> \\")
    print(f"      --npz_train_dir {data_path} \\")
    print(f"      --npz_val_dir <val_npz_dir> \\")
    if args.suffix:
        print(f"      --patch_token_key patch_tokens_{args.suffix} \\")
    print(f"      --output_dir <output_dir>")


if __name__ == "__main__":
    main()
