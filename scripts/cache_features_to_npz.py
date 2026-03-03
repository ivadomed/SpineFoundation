#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cache DINOv2 backbone features (masked average pooling) inside NPZ patches.

Run once before training with use_feature_caching=true to avoid re-running
the backbone at every training restart:

    python cache_features_to_npz.py \
        --data-dir /path/to/patches_RSNA_raw_with_mask_scs \
        --model-name /path/to/curia/snapshot \
        --batch-size 32

Each NPZ file gains a "features" key of shape (hidden_size,) float32.
Files that already have the "features" key are skipped (idempotent).
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoImageProcessor, Dinov2Model

# ── Constants (must match classification_hf/dataset.py) ──────────────────────
_DINO_PATCH = 14
_DILATE_R   = _DINO_PATCH // 2 + 1  # = 8


def resize_mask(mask_np: np.ndarray, target_size: int) -> torch.Tensor:
    """Explicit coordinate mapping + dilation (see classification_hf/dataset.py)."""
    H, W = mask_np.shape
    t = torch.zeros(1, 1, target_size, target_size)
    ys, xs = np.where(mask_np > 0)
    for y, x in zip(ys, xs):
        y_out = min(int(y * target_size / H), target_size - 1)
        x_out = min(int(x * target_size / W), target_size - 1)
        t[0, 0, y_out, x_out] = 1.0
    t = F.max_pool2d(t, kernel_size=2 * _DILATE_R + 1, stride=1, padding=_DILATE_R)
    return t  # (1, 1, target_size, target_size)


def masked_avg_pool(last_hidden_state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Masked average pooling over DINOv2 patch tokens.
    last_hidden_state : (B, 1+N, D)   — CLS at index 0
    mask              : (B, 1, S, S)  — binary, already resized + dilated
    Returns           : (B, D)
    """
    patch_tokens = last_hidden_state[:, 1:, :]          # (B, N, D)
    B, N, D = patch_tokens.shape
    grid = int(N ** 0.5)                                # e.g. 36 for 512/14

    mask_pooled = F.max_pool2d(mask.float(), kernel_size=_DINO_PATCH, stride=_DINO_PATCH)
    mask_flat   = mask_pooled.view(B, grid * grid).unsqueeze(-1)  # (B, N, 1)
    total       = mask_flat.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return (patch_tokens * mask_flat).sum(dim=1) / total.squeeze(1)  # (B, D)


def collect_npz_paths(data_dir: Path) -> list[Path]:
    """Collect all *.npz files under data_dir (one class sub-dir per class)."""
    paths = sorted(data_dir.rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No NPZ files found under {data_dir}")
    return paths


@torch.no_grad()
def process_batch(
    paths: list[Path],
    backbone: Dinov2Model,
    processor,
    device: torch.device,
    crop_size: int,
) -> dict[Path, np.ndarray]:
    """Run backbone + masked avg pool on a batch; return {path: features_np}."""
    images, masks, valid_paths = [], [], []

    for p in paths:
        d = np.load(p)
        images.append(d["slice"].astype(np.float32))
        if "mask" in d:
            masks.append(resize_mask(d["mask"], crop_size))  # (1, 1, S, S)
        else:
            masks.append(None)
        valid_paths.append(p)

    processed = processor(images, return_tensors="pt")
    pv = processed["pixel_values"].to(device)

    outputs = backbone(pixel_values=pv, output_hidden_states=False)

    if all(m is not None for m in masks):
        mask_batch = torch.cat(masks, dim=0).to(device)  # (B, 1, S, S)
        features = masked_avg_pool(outputs.last_hidden_state, mask_batch)
    else:
        # Fallback to CLS token when mask is absent
        features = outputs.last_hidden_state[:, 0]

    features_np = features.cpu().float().numpy()  # (B, D)
    return {p: features_np[i] for i, p in enumerate(valid_paths)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir",   required=True,
                    help="Root directory containing class sub-dirs with NPZ files.")
    ap.add_argument("--model-name", required=True,
                    help="Path or HF repo of the curia snapshot "
                         "(same as model.model_name in config).")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--skip-existing", action="store_true", default=True,
                    help="Skip NPZ files that already contain 'features' (default: on).")
    ap.add_argument("--force", action="store_true",
                    help="Recompute and overwrite existing 'features' entries.")
    args = ap.parse_args()

    data_dir   = Path(args.data_dir)
    model_name = args.model_name
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skip       = args.skip_existing and not args.force

    print(f"Loading backbone from {model_name} ...")
    backbone  = Dinov2Model.from_pretrained(model_name, trust_remote_code=True)
    backbone  = backbone.to(device).eval()
    processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
    crop_size = processor.crop_size
    print(f"  hidden_size={backbone.config.hidden_size}  crop_size={crop_size}  device={device}")

    all_paths = collect_npz_paths(data_dir)
    print(f"Found {len(all_paths)} NPZ files under {data_dir}")

    if skip:
        todo = [p for p in all_paths if "features" not in np.load(p).files]
        print(f"  {len(all_paths) - len(todo)} already cached, {len(todo)} to process.")
    else:
        todo = all_paths
        print(f"  --force: recomputing all {len(todo)} files.")

    if not todo:
        print("Nothing to do.")
        return

    n_saved = 0
    for i in tqdm(range(0, len(todo), args.batch_size), desc="Caching features"):
        batch_paths = todo[i : i + args.batch_size]
        feat_map = process_batch(batch_paths, backbone, processor, device, crop_size)

        for p, feat in feat_map.items():
            d = np.load(p)
            data = {k: d[k] for k in d.files}  # preserve slice, mask, …
            data["features"] = feat.astype(np.float32)
            np.savez_compressed(p, **data)
            n_saved += 1

    print(f"\nDone. Features cached in {n_saved}/{len(all_paths)} files.")
    print(f"Feature shape: ({backbone.config.hidden_size},) float32")


if __name__ == "__main__":
    main()
