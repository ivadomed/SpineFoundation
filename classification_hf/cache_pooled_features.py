"""
Pre-compute masked-avg-pooled features for a dataset directory and save them
to a single .pt file.  Run once per (data_dir, backbone) combination.

Output: ~/.cache/classification_hf/pooled_features_{data_dir_name}[_{suffix}].pt
    {
        "features": FloatTensor (N_samples, D),
        "labels":   LongTensor  (N_samples,),
        "paths":    list[str]   (N_samples,),
    }

── Mode 1: NPZ patch_tokens (no backbone needed, CPU, fast) ──────────────────
Used when --model_name is NOT provided.  Reads pre-cached patch_tokens from
each NPZ and applies masked-avg-pooling in parallel threads.

    python -m classification_hf.cache_pooled_features \\
        --data_dir /path/to/patches_RSNA_nfn --num_workers 16

── Mode 2: run a backbone (GPU, for custom encoders) ─────────────────────────
Used when --model_name IS provided.  Loads images from NPZ (slice key),
runs the frozen Dinov2Model, then applies masked-avg-pooling.

    python -m classification_hf.cache_pooled_features \\
        --data_dir /path/to/patches_RSNA_nfn \\
        --model_name /path/to/iter_416000 \\
        --processor_name /path/to/curia_snapshot \\
        --cache_suffix custom --batch_size 64

── Cache all three tasks for a custom backbone: ──────────────────────────────
    for TASK in nfn scs ss; do
        python -m classification_hf.cache_pooled_features \\
            --data_dir /home/.../data/RSNA_patches_$TASK \\
            --model_name /home/.../teacher_checkpoints/iter_416000 \\
            --processor_name /home/.../.cache/huggingface/hub/models--raidium--curia/snapshots/... \\
            --cache_suffix custom
    done
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
import time

from tqdm import tqdm

import numpy as np
from scipy.ndimage import maximum_filter, zoom
import torch
import torch.nn.functional as F

_DINO_PATCH = 16
_NPZ_EXT = ".npz"
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def resize_mask(mask_np: np.ndarray, target_size: int) -> torch.Tensor:
    t = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    t = F.interpolate(t, size=(target_size, target_size),
                      mode="bilinear", align_corners=False, antialias=True)
    mask = t > 0.5
    if mask.sum() == 0:
        mask = t > (t.max() * 0.5)
    return mask.float()  # (1, 1, target_size, target_size)


def pool_tokens(tokens: torch.Tensor, mask_t: torch.Tensor) -> torch.Tensor:
    """tokens: (N, D), mask_t: (1, 1, S, S) → (D,)"""
    N = tokens.shape[0]
    mask_pooled = F.max_pool2d(mask_t.float(), kernel_size=_DINO_PATCH, stride=_DINO_PATCH)
    mask_flat = mask_pooled.view(1, N)
    weights = mask_flat.unsqueeze(-1)
    total = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return ((tokens.unsqueeze(0) * weights).sum(dim=1) / total.squeeze(1)).squeeze(0)


# ── Mode 1: read patch_tokens from NPZ (CPU, multiprocess) ───────────────────

def _process_one_npz(args: tuple) -> np.ndarray:
    """Worker (runs in subprocess): load one NPZ, return pooled feature (D,).

    args = (path, token_key, use_mask)
    use_mask=False: mean-pool all tokens (used when tokens were extracted from
    a pre-cropped slice so the entire crop is the ROI).
    """
    path, token_key, use_mask = args
    d = np.load(path)
    if token_key not in d:
        raise ValueError(f"No '{token_key}' key in {path}")
    arr = d[token_key]

    # CLS token or any pre-pooled vector: already (D,), return as-is
    if arr.ndim == 1:
        return arr.astype(np.float32)

    tokens = arr.astype(np.float32)   # (N, D)
    N, D = tokens.shape
    grid = int(N ** 0.5)
    token_crop_size = grid * _DINO_PATCH

    if not use_mask or "mask" not in d:
        return tokens.mean(axis=0)

    mask = d["mask"].astype(np.float32)
    if mask.shape[0] != token_crop_size or mask.shape[1] != token_crop_size:
        scale = (token_crop_size / mask.shape[0], token_crop_size / mask.shape[1])
        mask = zoom(mask, scale, order=1)

    mask_bin = mask > 0.5
    if not mask_bin.any():
        mask_bin = mask > (mask.max() * 0.5)

    mask_grid = mask_bin.reshape(grid, _DINO_PATCH, grid, _DINO_PATCH)
    weights = mask_grid.any(axis=(1, 3)).astype(np.float32).ravel()  # (N,)

    total = weights.sum()
    if total < 1e-6:
        return tokens.mean(axis=0)
    return (tokens * weights[:, None]).sum(axis=0) / total


def _run_npz_mode(all_paths, all_labels, num_workers, out_path,
                  token_key: str = "patch_tokens", use_mask: bool = True):
    n = len(all_paths)
    pool_mode = "masked avg" if use_mask else "mean (no mask)"
    print(f"Mode            : NPZ {token_key} (CPU, {num_workers} processes, pooling={pool_mode})")
    args = [(p, token_key, use_mask) for p in all_paths]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        ordered = list(tqdm(
            executor.map(_process_one_npz, args, chunksize=32),
            total=n, desc="Pooling features", unit="ex",
        ))
    features = torch.from_numpy(np.stack(ordered))
    return features, time.time() - t0


# ── Mode 2: run backbone on GPU ───────────────────────────────────────────────

import queue
import threading

def _get_crop_size(processor) -> int:
    cs = processor.crop_size
    if isinstance(cs, dict):
        return cs["height"]
    return int(cs)


def _load_slice_and_mask(args_tuple):
    path, crop_size = args_tuple
    d = np.load(path)
    img = d["slice"].astype(np.float32)
    mask_t = resize_mask(d["mask"], crop_size) if "mask" in d else None
    return img, mask_t


def _run_backbone_mode(all_paths, all_labels, model_name,
                       processor_name, batch_size, out_path, num_readers=16):
    from transformers import AutoImageProcessor, Dinov2Model

    print(f"Mode            : backbone inference (GPU)")
    print(f"Backbone        : {model_name}")
    print(f"Processor       : {processor_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device          : {device}  (amp={use_amp})")

    processor = AutoImageProcessor.from_pretrained(processor_name, trust_remote_code=True)
    crop_size = _get_crop_size(processor)

    backbone = Dinov2Model.from_pretrained(model_name, trust_remote_code=True)
    backbone.to(device)
    backbone.eval()

    n = len(all_paths)
    features_list = []
    t0 = time.time()

    # ── Prefetch pipeline: CPU load+preprocess runs ahead of GPU ──────────────
    _SENTINEL = object()
    prefetch_q: queue.Queue = queue.Queue(maxsize=2)

    def _feeder():
        with ThreadPoolExecutor(max_workers=num_readers) as reader_pool:
            for start in range(0, n, batch_size):
                batch_paths = all_paths[start:start + batch_size]
                loaded = list(reader_pool.map(
                    _load_slice_and_mask,
                    [(p, crop_size) for p in batch_paths]
                ))
                images_np    = [x[0] for x in loaded]
                mask_tensors = [x[1] for x in loaded]
                processed    = processor(images_np, return_tensors="pt")
                pixel_values = processed["pixel_values"]
                if use_amp:
                    pixel_values = pixel_values.pin_memory()
                prefetch_q.put((pixel_values, mask_tensors, len(batch_paths)))
        prefetch_q.put(_SENTINEL)

    threading.Thread(target=_feeder, daemon=True).start()

    with tqdm(total=n, desc="Extracting features", unit="ex") as pbar:
        while True:
            item = prefetch_q.get()
            if item is _SENTINEL:
                break
            pixel_values, mask_tensors, batch_n = item

            with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = backbone(
                    pixel_values=pixel_values.to(device, non_blocking=True),
                    output_hidden_states=False,
                )

            patch_tokens = outputs.last_hidden_state[:, 1:, :].float()  # skip CLS → (B, N, D)

            if all(m is not None for m in mask_tensors):
                mask_batch = torch.cat(mask_tensors, dim=0).to(device, non_blocking=True)
                B, N, D = patch_tokens.shape
                grid = int(N ** 0.5)
                actual_crop = grid * _DINO_PATCH
                if mask_batch.shape[-1] != actual_crop:
                    mask_batch = F.interpolate(
                        mask_batch.float(), size=(actual_crop, actual_crop), mode="nearest"
                    )
                mask_pooled = F.max_pool2d(
                    mask_batch.float(), kernel_size=_DINO_PATCH, stride=_DINO_PATCH
                )  # (B, 1, grid, grid)
                mask_flat = mask_pooled.view(B, N)
                weights = mask_flat.unsqueeze(-1)
                total = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
                pooled = (patch_tokens * weights).sum(dim=1) / total.squeeze(1)  # (B, D)
            else:
                pooled = patch_tokens.mean(dim=1)

            features_list.append(pooled.cpu())
            pbar.update(batch_n)

    features = torch.cat(features_list, dim=0)
    return features, time.time() - t0


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",        required=True)
    parser.add_argument("--cache_suffix",    type=str, default="",
                        help="Suffix to distinguish caches from different backbones, "
                             "e.g. --cache_suffix custom  →  pooled_features_custom_dil8.pt")
    # Mode 2 options (backbone inference)
    parser.add_argument("--model_name",      type=str, default=None,
                        help="Path to a Dinov2 backbone checkpoint. If not set, reads "
                             "patch_tokens directly from NPZ files (mode 1, no GPU needed).")
    parser.add_argument("--processor_name",  type=str, default=None,
                        help="Path to AutoImageProcessor (defaults to --model_name). "
                             "Useful when the checkpoint has no preprocessor_config.json.")
    parser.add_argument("--batch_size",      type=int, default=64,
                        help="Batch size for backbone inference (mode 2 only, default: 64)")
    # Mode 1 options (NPZ patch_tokens)
    parser.add_argument("--token_key",       type=str, default="patch_tokens",
                        help="NPZ key to read for mode 1 (default: patch_tokens). "
                             "Use patch_tokens_custom for a custom backbone cached via "
                             "cache_patch_tokens.py --suffix custom.")
    parser.add_argument("--num_workers",     type=int, default=16,
                        help="Parallel threads for NPZ I/O (default: 16)")
    parser.add_argument("--no_mask",         action="store_true",
                        help="Mean-pool ALL patch tokens instead of masked average pooling. "
                             "Use this when patch_tokens were extracted from a pre-cropped "
                             "slice (e.g. via cache_patch_tokens.py --crop_cm 4), so the "
                             "original mask coordinates are no longer valid.")
    parser.add_argument("--cache_dir",       type=str, default=None,
                        help="Directory to store the output .pt file "
                             "(default: ~/.cache/classification_hf/).")
    args = parser.parse_args()

    data_path   = Path(args.data_dir)
    suffix_part = f"_{args.cache_suffix}" if args.cache_suffix else ""
    cache_root  = Path(args.cache_dir) if args.cache_dir else Path.home() / ".cache" / "classification_hf"
    cache_root.mkdir(parents=True, exist_ok=True)
    out_path    = cache_root / f"pooled_features_{data_path.name}{suffix_part}.pt"

    if out_path.exists():
        print(f"Cache already exists: {out_path}  (delete to regenerate)")
        return

    class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class subdirectories found in {data_path}")

    all_paths, all_labels = [], []
    for label, class_dir in enumerate(class_dirs):
        files = sorted([
            f for f in class_dir.iterdir()
            if f.suffix.lower() in {_NPZ_EXT} | _IMG_EXTS
            and not f.name.startswith("tmp")
        ])
        for f in files:
            all_paths.append(str(f))
            all_labels.append(label)

    n = len(all_paths)
    use_mask = not args.no_mask
    print(f"Found {n} samples across {len(class_dirs)} classes in {data_path}")
    print(f"Mask pooling    : {'yes' if use_mask else 'no (mean over all tokens)'}")
    print(f"Output          : {out_path}")

    if args.model_name:
        processor_name = args.processor_name or args.model_name
        features, elapsed = _run_backbone_mode(
            all_paths, all_labels,
            args.model_name, processor_name, args.batch_size, out_path,
            num_readers=args.num_workers,
        )
    else:
        features, elapsed = _run_npz_mode(
            all_paths, all_labels, args.num_workers, out_path,
            token_key=args.token_key, use_mask=use_mask,
        )

    labels = torch.tensor(all_labels, dtype=torch.long)
    torch.save({"features": features, "labels": labels, "paths": all_paths}, out_path)

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nDone in {elapsed:.1f}s  →  {out_path}  ({size_mb:.1f} MB)")
    print(f"Shape: features={tuple(features.shape)}, labels={tuple(labels.shape)}")


if __name__ == "__main__":
    main()
