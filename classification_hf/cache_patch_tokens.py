"""
Run a Dinov2 backbone over a dataset directory and save the raw (non-pooled)
patch tokens back into each NPZ file under a suffixed key.

After running this script once, generate pooled_features_*.pt using
cache_pooled_features.py (mode 1, CPU only):

    python -m classification_hf.cache_patch_tokens \\
        --data_dir /path/to/RSNA_patches_nfn \\
        --model_name /path/to/custom_backbone/iter_416000 \\
        --processor_name /path/to/curia_snapshot \\
        --suffix custom

Then pool the features:

    python -m classification_hf.cache_pooled_features \\
        --data_dir /path/to/RSNA_patches_nfn \\
        --token_key patch_tokens_custom \\
        --cache_suffix custom

The NPZ files are updated in-place: all existing keys are preserved and
patch_tokens_{suffix} is added (or overwritten if it already exists).

Optimisations vs version naïve :
  - Pré-check parallèle (16 threads) au lieu de séquentiel
  - Lecture NPZ unique par fichier (pas de double chargement)
  - Écriture savez_compressed asynchrone (8 threads) pendant que le GPU
    traite le batch suivant → GPU idle ≈ 0%
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import os
from pathlib import Path
import shutil
import tempfile
import time
import zipfile

import numpy as np
import torch
from tqdm import tqdm

_NPZ_EXT  = ".npz"
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _fm_shortname(model_path: str) -> str:
    """Extract a short model name from a HuggingFace hub path or local path.

    Examples:
      .../models--raidium--curia/snapshots/abc/  → "curia"
      .../models--raidium--mricore/snapshots/abc/ → "mricore"
      /local/my_custom_model/                     → "my_custom_model"
    """
    parts = Path(model_path).parts
    for part in reversed(parts):
        if part.startswith("models--"):
            # models--org--name → name
            segments = part.split("--")
            if len(segments) >= 3:
                return segments[-1]
        if part not in {"snapshots", "blobs", "refs"} and not _looks_like_hash(part):
            return part
    return Path(model_path).name or "fm"


def _looks_like_hash(s: str) -> bool:
    return len(s) >= 8 and all(c in "0123456789abcdef" for c in s.lower())


def _has_key(path: str, key: str) -> bool:
    """Return True if the NPZ already contains 'key' (reads only the header)."""
    try:
        return key in np.load(path).files
    except Exception:
        return False


def _crop_around_mask(img: np.ndarray, mask: np.ndarray,
                      spacing_mm: float, crop_cm: float) -> np.ndarray:
    """Return a square crop of img centered on the mask centroid.

    crop_cm: desired side length in centimetres (e.g. 4.0 → 40 mm).
    Zero-pads if the centroid is too close to the image border.
    """
    crop_px = int(round(crop_cm * 10.0 / spacing_mm))
    half = crop_px // 2

    ys, xs = np.where(mask > 0)
    cy = int(ys[0]) if len(ys) > 0 else img.shape[0] // 2
    cx = int(xs[0]) if len(xs) > 0 else img.shape[1] // 2

    H, W = img.shape
    y0, y1 = cy - half, cy - half + crop_px
    x0, x1 = cx - half, cx - half + crop_px

    pad_top = max(0, -y0)
    pad_bot = max(0, y1 - H)
    pad_lft = max(0, -x0)
    pad_rgt = max(0, x1 - W)

    crop = img[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    if pad_top or pad_bot or pad_lft or pad_rgt:
        crop = np.pad(crop, ((pad_top, pad_bot), (pad_lft, pad_rgt)),
                      mode="constant", constant_values=0.0)
    return crop


def _load_npz_for_inference(path: str, crop_cm: float | None = None) -> tuple[str, np.ndarray]:
    """Load only the keys needed for inference (slice, mask, spacing_mm).

    Avoids decompressing existing patch_tokens arrays (~96ms/file saved).
    Returns (path, image_array).
    """
    d = np.load(path)
    img = d["slice"].astype(np.float32)

    if crop_cm is not None:
        if "spacing_mm" not in d.files:
            raise KeyError(
                f"'spacing_mm' not found in {path}. "
                "Run add_spacing_to_npz.py first."
            )
        spacing_mm = float(d["spacing_mm"])
        mask = d["mask"] if "mask" in d.files else np.zeros(img.shape, dtype=np.uint8)
        img = _crop_around_mask(img, mask, spacing_mm, crop_cm)

    return path, img


def _append_token_to_npz(path: str, token_key: str, tokens_np: np.ndarray) -> None:
    """Append a new array to an existing NPZ without rewriting existing keys.

    NPZ files are ZIP archives. We copy the file and append the new .npy entry
    (uncompressed — DINOv2 float32 tokens are ~incompressible) instead of
    decompressing + recompressing the whole archive.

    Atomicity: write to a temp file beside the original, then os.replace().
    """
    buf = io.BytesIO()
    np.save(buf, tokens_np)
    npy_bytes = buf.getvalue()

    dirpath = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=dirpath, suffix=".npz")
    os.close(fd)
    try:
        shutil.copy2(path, tmp_path)
        with zipfile.ZipFile(tmp_path, mode="a", compression=zipfile.ZIP_STORED) as zf:
            # Remove stale entry if key already exists (overwrite mode)
            existing_names = zf.namelist()
            npy_name = f"{token_key}.npy"
            if npy_name in existing_names:
                # zipfile doesn't support deletion; fall back to full rewrite
                raise ValueError(f"_overwrite_needed:{token_key}")
            zf.writestr(npy_name, npy_bytes)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _save_npz_full(path: str, token_key: str, tokens_np: np.ndarray) -> None:
    """Full rewrite fallback (used only when overwriting an existing key)."""
    d = np.load(path)
    existing = {k: d[k] for k in d.files}
    existing[token_key] = tokens_np
    dirpath = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=dirpath, suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp_path, **existing)
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


def _save_npz(path: str, token_key: str, tokens_np: np.ndarray,
              overwrite: bool = False) -> None:
    """Write token array to NPZ: append mode (fast) or full rewrite (overwrite)."""
    if overwrite:
        _save_npz_full(path, token_key, tokens_np)
        return
    try:
        _append_token_to_npz(path, token_key, tokens_np)
    except ValueError as e:
        if "_overwrite_needed:" in str(e):
            _save_npz_full(path, token_key, tokens_np)
        else:
            raise


def _save_npz_multi(path: str, arrays: dict[str, np.ndarray],
                    overwrite: bool = False) -> None:
    """Write multiple arrays to an NPZ in a single atomic operation.

    Avoids race conditions when saving several keys for the same file
    (e.g. patch_tokens + cls_token in the same inference pass).
    """
    if overwrite:
        d = np.load(path)
        existing = {k: d[k] for k in d.files}
        existing.update(arrays)
        dirpath = os.path.dirname(path)
        fd, tmp = tempfile.mkstemp(dir=dirpath, suffix=".npz")
        os.close(fd)
        try:
            np.savez_compressed(tmp, **existing)
            os.replace(tmp, path)
        except Exception:
            os.unlink(tmp)
            raise
        return

    # Fast path: append all new keys as uncompressed entries in one pass
    dirpath = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=dirpath, suffix=".npz")
    os.close(fd)
    try:
        shutil.copy2(path, tmp)
        with zipfile.ZipFile(tmp, mode="a", compression=zipfile.ZIP_STORED) as zf:
            existing_names = set(zf.namelist())
            needs_full_rewrite = any(f"{k}.npy" in existing_names for k in arrays)
            if needs_full_rewrite:
                raise ValueError("_overwrite_needed")
            for k, arr in arrays.items():
                buf = io.BytesIO()
                np.save(buf, arr)
                zf.writestr(f"{k}.npy", buf.getvalue())
        os.replace(tmp, path)
    except ValueError as e:
        if os.path.exists(tmp):
            os.unlink(tmp)
        if "_overwrite_needed" in str(e):
            _save_npz_multi(path, arrays, overwrite=True)
        else:
            raise
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",       required=True,
                        help="Root directory with class sub-dirs containing NPZ files.")
    parser.add_argument("--model_name",     required=True,
                        help="Path to a Dinov2 backbone checkpoint.")
    parser.add_argument("--processor_name", default=None,
                        help="Path to AutoImageProcessor (defaults to --model_name). "
                             "Useful when the checkpoint has no preprocessor_config.json.")
    parser.add_argument("--suffix",         default=None,
                        help="Optional extra suffix appended after the FM name. "
                             "Key = patch_tokens_{fm}_{suffix} (or patch_tokens_{fm} if omitted). "
                             "E.g. --suffix crop4cm → patch_tokens_curia_crop4cm")
    parser.add_argument("--batch_size",     type=int, default=64)
    parser.add_argument("--num_readers",    type=int, default=16,
                        help="Threads for parallel pre-check and NPZ reads (default: 16)")
    parser.add_argument("--num_writers",    type=int, default=8,
                        help="Threads for async NPZ writes (default: 8)")
    parser.add_argument("--overwrite",      action="store_true",
                        help="Re-compute and overwrite existing patch_tokens_{suffix} keys.")
    parser.add_argument("--crop_cm",        type=float, default=None,
                        help="If set, crop each slice to crop_cm×crop_cm centimetres centred "
                             "on the mask centroid before running the backbone. Requires "
                             "'spacing_mm' in every NPZ (produced by RSNAextractor.py).")
    args = parser.parse_args()

    data_path      = Path(args.data_dir)
    fm_name        = _fm_shortname(args.model_name)
    token_key      = f"patch_tokens_{fm_name}_{args.suffix}" if args.suffix else f"patch_tokens_{fm_name}"
    processor_name = args.processor_name or args.model_name

    # ── Collect all NPZ paths ─────────────────────────────────────────────────
    class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class subdirectories found in {data_path}")

    all_paths = []
    bad_files = []
    for class_dir in class_dirs:
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() == _NPZ_EXT:
                try:
                    np.load(str(f), allow_pickle=False).files
                    all_paths.append(str(f))
                except Exception:
                    bad_files.append(str(f))
    if bad_files:
        print(f"[WARN] {len(bad_files)} NPZ invalides ignorés (fichiers corrompus ou temp) :")
        for b in bad_files:
            print(f"  {b}")

    n = len(all_paths)
    print(f"Found {n} NPZ files across {len(class_dirs)} classes in {data_path}")

    # ── Parallel pre-check (reads only NPZ header, not the arrays) ────────────
    cls_key = token_key.replace("patch_tokens_", "cls_token_", 1)

    def _has_both_keys(path: str) -> bool:
        try:
            files = np.load(path, allow_pickle=False).files
            return token_key in files and cls_key in files
        except Exception:
            return False

    if not args.overwrite:
        print(f"Checking existing '{token_key}' + '{cls_key}' keys "
              f"({args.num_readers} threads)...", flush=True)
        pending = []
        with ThreadPoolExecutor(max_workers=args.num_readers) as ex:
            futures = {ex.submit(_has_both_keys, p): p for p in all_paths}
            for fut in tqdm(as_completed(futures), total=n,
                            desc="Pre-check", unit="file", leave=False):
                if not fut.result():
                    pending.append(futures[fut])
        skipped = n - len(pending)
        if skipped:
            print(f"Skipping {skipped} files already containing both keys "
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
    if args.crop_cm is not None:
        print(f"Crop             : {args.crop_cm} cm × {args.crop_cm} cm (centred on mask)")
    else:
        print(f"Crop             : none (full slice)")

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
    # We wait for the previous batch's writes AFTER the current GPU batch,
    # so writes(N-1) have the full GPU(N) window to finish.
    t0 = time.time()
    errors = []
    overwrite = args.overwrite
    crop_cm   = args.crop_cm
    prev_futures: list = []   # writes submitted for batch N-1

    def _wait_for(futures: list) -> None:
        for f in futures:
            try:
                f.result()
            except Exception as e:
                errors.append(str(e))

    from functools import partial as _partial
    _load = _partial(_load_npz_for_inference, crop_cm=crop_cm)

    with ThreadPoolExecutor(max_workers=args.num_readers) as reader_pool, \
         ThreadPoolExecutor(max_workers=args.num_writers) as writer_pool:
        with tqdm(total=len(all_paths), desc="Caching patch_tokens", unit="ex") as pbar:
            for start in range(0, len(all_paths), args.batch_size):
                batch_paths = all_paths[start : start + args.batch_size]

                # Parallel reads (slice+mask+spacing only — skip existing patch_tokens)
                loaded    = list(reader_pool.map(_load, batch_paths))
                images_np = [x[1] for x in loaded]

                # Parallel preprocessing (resize + normalize per image in reader threads)
                def _preprocess_one(img):
                    return processor([img], return_tensors="pt")["pixel_values"][0]
                pixel_values = torch.stack(
                    list(reader_pool.map(_preprocess_one, images_np))
                ).to(device)

                with torch.no_grad():
                    outputs = backbone(pixel_values=pixel_values,
                                       output_hidden_states=False)

                hidden = outputs.last_hidden_state.cpu().float()
                cls_batch         = hidden[:, 0, :]    # (B, D)
                patch_tokens_batch = hidden[:, 1:, :]  # (B, N, D)

                # Wait for previous batch's writes (had the GPU window to complete)
                _wait_for(prev_futures)

                # Submit current batch's writes asynchronously — patch_tokens + cls_token
                # in a single atomic write per file to avoid race conditions
                prev_futures = [
                    writer_pool.submit(
                        _save_npz_multi, path,
                        {
                            token_key: patch_tokens_batch[i].numpy(),
                            cls_key:   cls_batch[i].numpy(),
                        },
                        overwrite,
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
    print(f"\nNext step: generate pooled .pt (CPU, fast):")
    print(f"  python -m classification_hf.cache_pooled_features \\")
    print(f"      --data_dir {data_path} \\")
    print(f"      --token_key {token_key} \\")
    print(f"      --cache_suffix {args.suffix}")


if __name__ == "__main__":
    main()
