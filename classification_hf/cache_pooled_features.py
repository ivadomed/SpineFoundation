"""
Pre-compute masked-avg-pooled features for a dataset directory and save them
to a single .pt file.  Run once per (data_dir, backbone, dilation_radius) combination.

Output: {data_dir}/pooled_features[_{suffix}]_dil{N}.pt
    {
        "features": FloatTensor (N_samples, D),
        "labels":   LongTensor  (N_samples,),
        "paths":    list[str]   (N_samples,),
    }

── Mode 1: NPZ patch_tokens (no backbone needed, CPU, fast) ──────────────────
Used when --model_name is NOT provided.  Reads pre-cached patch_tokens from
each NPZ and applies masked-avg-pooling in parallel threads.

    python -m classification_hf.cache_pooled_features \\
        --data_dir /path/to/patches_RSNA_raw_with_mask_nfn \\
        --dilation_radius 8 --num_workers 16

── Mode 2: run a backbone (GPU, for custom encoders) ─────────────────────────
Used when --model_name IS provided.  Loads images from NPZ (slice key),
runs the frozen Dinov2Model, then applies masked-avg-pooling.

    python -m classification_hf.cache_pooled_features \\
        --data_dir /path/to/patches_RSNA_raw_with_mask_nfn \\
        --model_name /path/to/iter_416000 \\
        --processor_name /path/to/curia_snapshot \\
        --cache_suffix custom \\
        --dilation_radius 8 --batch_size 64

── Cache all three tasks for a custom backbone: ──────────────────────────────
    for TASK in nfn scs ss; do
        python -m classification_hf.cache_pooled_features \\
            --data_dir /home/.../data/patches_RSNA_raw_with_mask_$TASK \\
            --model_name /home/.../outputs_curia/custom_curia_512/teacher_checkpoints/iter_416000 \\
            --processor_name /home/.../.cache/huggingface/hub/models--raidium--curia/snapshots/... \\
            --cache_suffix custom --dilation_radius 8
    done
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

_DINO_PATCH = 14
_NPZ_EXT = ".npz"
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def resize_mask(mask_np: np.ndarray, target_size: int, dilation_radius: int) -> torch.Tensor:
    t = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    t = F.interpolate(t, size=(target_size, target_size),
                      mode="bilinear", align_corners=False, antialias=True)
    mask = t > 0.5
    if mask.sum() == 0:
        mask = t > (t.max() * 0.5)
    t = mask.float()
    if dilation_radius > 0:
        t = F.max_pool2d(t, kernel_size=2 * dilation_radius + 1, stride=1, padding=dilation_radius)
    return t  # (1, 1, target_size, target_size)


def pool_tokens(tokens: torch.Tensor, mask_t: torch.Tensor) -> torch.Tensor:
    """tokens: (N, D), mask_t: (1, 1, S, S) → (D,)"""
    N = tokens.shape[0]
    mask_pooled = F.max_pool2d(mask_t.float(), kernel_size=_DINO_PATCH, stride=_DINO_PATCH)
    mask_flat = mask_pooled.view(1, N)
    weights = mask_flat.unsqueeze(-1)
    total = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return ((tokens.unsqueeze(0) * weights).sum(dim=1) / total.squeeze(1)).squeeze(0)


# ── Mode 1: read patch_tokens from NPZ (CPU) ──────────────────────────────────

def _process_one_npz(path: str, dilation_radius: int, token_key: str = "patch_tokens") -> torch.Tensor:
    """Load one NPZ and return the pooled feature vector (D,)."""
    d = np.load(path)
    if token_key not in d:
        raise ValueError(f"No '{token_key}' key in {path}")
    tokens = torch.from_numpy(d[token_key].copy())
    grid = int(tokens.shape[0] ** 0.5)
    token_crop_size = grid * _DINO_PATCH
    if "mask" in d:
        mask_t = resize_mask(d["mask"], token_crop_size, dilation_radius)
        return pool_tokens(tokens, mask_t)
    return tokens.mean(dim=0)


def _run_npz_mode(all_paths, all_labels, dilation_radius, num_workers, out_path,
                  token_key: str = "patch_tokens"):
    n = len(all_paths)
    print(f"Mode            : NPZ {token_key} (CPU, {num_workers} threads)")
    results = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_process_one_npz, path, dilation_radius, token_key): i
            for i, path in enumerate(all_paths)
        }
        with tqdm(total=n, desc="Pooling features", unit="ex") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
                pbar.update(1)
    features = torch.stack([results[i] for i in range(n)])
    return features, time.time() - t0


# ── Mode 2: run backbone on GPU ───────────────────────────────────────────────

def _get_crop_size(processor) -> int:
    cs = processor.crop_size
    if isinstance(cs, dict):
        return cs["height"]
    return int(cs)


def _load_slice_and_mask(args_tuple):
    path, crop_size, dilation_radius = args_tuple
    d = np.load(path)
    img = d["slice"].astype(np.float32)
    mask_t = resize_mask(d["mask"], crop_size, dilation_radius) if "mask" in d else None
    return img, mask_t


def _run_backbone_mode(all_paths, all_labels, dilation_radius, model_name,
                       processor_name, batch_size, out_path, num_readers=16):
    from transformers import AutoImageProcessor, Dinov2Model

    print(f"Mode            : backbone inference (GPU)")
    print(f"Backbone        : {model_name}")
    print(f"Processor       : {processor_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device          : {device}")

    processor = AutoImageProcessor.from_pretrained(processor_name, trust_remote_code=True)
    crop_size = _get_crop_size(processor)

    backbone = Dinov2Model.from_pretrained(model_name, trust_remote_code=True)
    backbone.to(device)
    backbone.eval()

    n = len(all_paths)
    features_list = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=num_readers) as reader_pool:
        with tqdm(total=n, desc="Extracting features", unit="ex") as pbar:
            for start in range(0, n, batch_size):
                batch_paths = all_paths[start:start + batch_size]

                # Parallel reads — overlaps with GPU from previous batch
                loaded = list(reader_pool.map(
                    _load_slice_and_mask,
                    [(p, crop_size, dilation_radius) for p in batch_paths]
                ))
                images_np    = [x[0] for x in loaded]
                mask_tensors = [x[1] for x in loaded]

                processed = processor(images_np, return_tensors="pt")
                pixel_values = processed["pixel_values"].to(device)

                with torch.no_grad():
                    outputs = backbone(pixel_values=pixel_values, output_hidden_states=False)

                patch_tokens = outputs.last_hidden_state[:, 1:, :]  # skip CLS → (B, N, D)

                if all(m is not None for m in mask_tensors):
                    mask_batch = torch.cat(mask_tensors, dim=0).to(device)  # (B, 1, S, S)
                    mask_pooled = F.max_pool2d(
                        mask_batch.float(), kernel_size=_DINO_PATCH, stride=_DINO_PATCH
                    )  # (B, 1, grid, grid)
                    B, N, D = patch_tokens.shape
                    mask_flat = mask_pooled.view(B, N)
                    weights = mask_flat.unsqueeze(-1)
                    total = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
                    pooled = (patch_tokens * weights).sum(dim=1) / total.squeeze(1)  # (B, D)
                else:
                    pooled = patch_tokens.mean(dim=1)

                features_list.append(pooled.cpu())
                pbar.update(len(batch_paths))

    features = torch.cat(features_list, dim=0)
    return features, time.time() - t0


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",        required=True)
    parser.add_argument("--dilation_radius", type=int, default=8)
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
    args = parser.parse_args()

    data_path   = Path(args.data_dir)
    suffix_part = f"_{args.cache_suffix}" if args.cache_suffix else ""
    out_path    = data_path / f"pooled_features{suffix_part}_dil{args.dilation_radius}.pt"

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
    print(f"Found {n} samples across {len(class_dirs)} classes in {data_path}")
    print(f"Dilation radius : {args.dilation_radius}")
    print(f"Output          : {out_path}")

    if args.model_name:
        processor_name = args.processor_name or args.model_name
        features, elapsed = _run_backbone_mode(
            all_paths, all_labels, args.dilation_radius,
            args.model_name, processor_name, args.batch_size, out_path,
            num_readers=args.num_workers,
        )
    else:
        features, elapsed = _run_npz_mode(
            all_paths, all_labels, args.dilation_radius, args.num_workers, out_path,
            token_key=args.token_key,
        )

    labels = torch.tensor(all_labels, dtype=torch.long)
    torch.save({"features": features, "labels": labels, "paths": all_paths}, out_path)

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nDone in {elapsed:.1f}s  →  {out_path}  ({size_mb:.1f} MB)")
    print(f"Shape: features={tuple(features.shape)}, labels={tuple(labels.shape)}")


if __name__ == "__main__":
    main()
