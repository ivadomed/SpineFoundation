"""
Extract patch tokens from MRI-CORE (SAM ViT-B) and save them into each NPZ
file under the key 'patch_tokens_mricore'.

Usage:
    python -m classification_hf.cache_patch_tokens_mricore \\
        --data_dir /path/to/patches_RSNA_raw_with_mask_nfn \\
        --checkpoint /path/to/MRI_CORE_vitb.pth \\
        --mri_foundation_dir /path/to/mri_foundation

Then pool with the existing cache_pooled_features.py (mode 1, CPU):
    python -m classification_hf.cache_pooled_features \\
        --data_dir /path/to/patches_RSNA_raw_with_mask_nfn \\
        --token_key patch_tokens_mricore \\
        --cache_suffix mricore \\
        --dilation_radius 8

MRI-CORE image encoder:
  - Input : FloatTensor (B, 3, 1024, 1024) normalized to [0, 1]
  - Output: spatial feature map (B, 256, 64, 64)
  - Reshaped to patch tokens (B, 4096, 256) before saving,
    matching the (N, D) layout expected by cache_pooled_features.py
    (grid=64, patch_size=16, token_crop_size=1024).
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_NPZ_EXT  = ".npz"
_TOKEN_KEY = "patch_tokens_mricore"
_IMAGE_SIZE = 1024   # MRI-CORE input resolution


# ── NPZ helpers (same as cache_patch_tokens.py) ───────────────────────────────

def _has_key(path: str, key: str) -> bool:
    try:
        return key in np.load(path).files
    except Exception:
        return False


def _load_npz(path: str, skip_key: str | None = None) -> tuple[str, dict, np.ndarray]:
    d = np.load(path)
    existing = {k: d[k] for k in d.files if k != skip_key}
    img = existing["slice"].astype(np.float32)
    return path, existing, img


def _save_npz(path: str, existing: dict, token_key: str, tokens_np: np.ndarray) -> None:
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


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_batch(images_np: list[np.ndarray], image_size: int = _IMAGE_SIZE,
                     device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Convert a list of float32 2D arrays (H, W) to a (B, 3, image_size, image_size)
    FloatTensor normalized to [0, 1].
    """
    tensors = []
    for img in images_np:
        # Min-max normalize to [0, 1]
        lo, hi = img.min(), img.max()
        if hi > lo:
            img = (img - lo) / (hi - lo)
        else:
            img = np.zeros_like(img)

        t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)

        # Resize to model input size on GPU
        if t.shape[-2] != image_size or t.shape[-1] != image_size:
            t = F.interpolate(t, size=(image_size, image_size),
                              mode="bilinear", align_corners=False)

        # Grayscale → 3 channels
        t = t.repeat(1, 3, 1, 1)   # (1, 3, image_size, image_size)
        tensors.append(t)

    return torch.cat(tensors, dim=0)  # (B, 3, image_size, image_size) on device


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",           required=True,
                        help="Root directory with class sub-dirs containing NPZ files.")
    parser.add_argument("--checkpoint",         required=True,
                        help="Path to MRI_CORE_vitb.pth.")
    parser.add_argument("--mri_foundation_dir", required=True,
                        help="Path to the cloned mazurowski-lab/mri_foundation repo.")
    parser.add_argument("--batch_size",         type=int, default=8,
                        help="Batch size (default: 8 — 1024px images are large).")
    parser.add_argument("--num_readers",        type=int, default=16)
    parser.add_argument("--num_writers",        type=int, default=8)
    parser.add_argument("--overwrite",          action="store_true")
    args = parser.parse_args()

    # Add mri_foundation to path so we can import models.sam and cfg
    sys.path.insert(0, str(Path(args.mri_foundation_dir).resolve()))
    from models.sam import sam_model_registry
    import cfg as mricore_cfg

    data_path  = Path(args.data_dir)
    token_key  = _TOKEN_KEY

    # ── Collect NPZ paths ─────────────────────────────────────────────────────
    class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class subdirectories found in {data_path}")

    all_paths = []
    for class_dir in class_dirs:
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() == _NPZ_EXT:
                all_paths.append(str(f))

    n = len(all_paths)
    print(f"Found {n} NPZ files across {len(class_dirs)} classes in {data_path}")

    # ── Pre-check ─────────────────────────────────────────────────────────────
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
            print(f"Skipping {skipped} files already containing '{token_key}'")
        all_paths = pending

    if not all_paths:
        print("Nothing to do.")
        return

    print(f"To process       : {len(all_paths)} files")
    print(f"Checkpoint       : {args.checkpoint}")
    print(f"Token key        : {token_key}")
    print(f"Batch size       : {args.batch_size}")

    # ── Load MRI-CORE model ───────────────────────────────────────────────────
    import sys as _sys
    _saved_argv, _sys.argv = _sys.argv, _sys.argv[:1]
    mricore_args = mricore_cfg.parse_args()
    _sys.argv = _saved_argv
    mricore_args.if_encoder_adapter      = False
    mricore_args.if_mask_decoder_adapter = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device           : {device}\n")

    model = sam_model_registry["vit_b"](
        mricore_args,
        checkpoint=args.checkpoint,
        num_classes=mricore_args.num_cls,
        image_size=mricore_args.image_size,
        pretrained_sam=False,
    )
    model.to(device)
    model.eval()

    encoder = model.image_encoder  # (B, 3, 1024, 1024) → (B, 256, 64, 64)

    # Bypass the randomly-initialized neck: hook the last ViT block to
    # capture the 768-dim tokens before the neck projection.
    _vit_out = {}
    def _hook(module, input, output):
        _vit_out['x'] = output  # (B, H, W, 768)
    _handle = encoder.blocks[-1].register_forward_hook(_hook)

    # ── Process in batches ────────────────────────────────────────────────────
    t0 = time.time()
    errors       = []
    skip_key     = token_key if args.overwrite else None
    prev_futures: list = []

    def _wait_for(futures: list) -> None:
        for f in futures:
            try:
                f.result()
            except Exception as e:
                errors.append(str(e))

    with ThreadPoolExecutor(max_workers=args.num_readers) as reader_pool, \
         ThreadPoolExecutor(max_workers=args.num_writers) as writer_pool:
        with tqdm(total=len(all_paths), desc="Caching patch_tokens_mricore", unit="ex") as pbar:
            for start in range(0, len(all_paths), args.batch_size):
                batch_paths = all_paths[start : start + args.batch_size]

                loaded         = list(reader_pool.map(
                    lambda p: _load_npz(p, skip_key), batch_paths
                ))
                batch_existing = [x[1] for x in loaded]
                images_np      = [x[2] for x in loaded]

                # Preprocess: (B, 3, 1024, 1024), float [0, 1] — resize on GPU
                pixel_values = preprocess_batch(images_np, device=device)

                with torch.no_grad(), torch.amp.autocast(device_type=device.type):
                    encoder(pixel_values)  # triggers hook, output discarded

                # Use pre-neck 768d tokens captured by hook: (B, H, W, 768)
                x = _vit_out['x']
                B, H, W, D = x.shape
                # (B, H, W, D) → (B, H*W, D) = (B, 4096, 768)
                patch_tokens_batch = x.reshape(B, H * W, D)
                patch_tokens_batch = patch_tokens_batch.cpu().half()  # float16 — 2× smaller/faster

                _wait_for(prev_futures)

                prev_futures = [
                    writer_pool.submit(
                        _save_npz, path, batch_existing[i], token_key,
                        patch_tokens_batch[i].numpy()
                    )
                    for i, path in enumerate(batch_paths)
                ]

                pbar.update(len(batch_paths))

        if prev_futures:
            print(f"Waiting for {len(prev_futures)} final write(s)...", flush=True)
            _wait_for(prev_futures)

    if errors:
        print(f"\n[WARN] {len(errors)} write errors:")
        for e in errors[:5]:
            print(f"  {e}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s — '{token_key}' added to {len(all_paths)} NPZ files.")
    print(f"\nNext step: pool features for a given dilation radius:")
    print(f"  python -m classification_hf.cache_pooled_features \\")
    print(f"      --data_dir {data_path} \\")
    print(f"      --token_key {token_key} \\")
    print(f"      --cache_suffix mricore \\")
    print(f"      --dilation_radius 8")


if __name__ == "__main__":
    main()
