#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cache DINOv2 backbone raw patch tokens inside NPZ patches.

Saves patch tokens (N, D) per slice — masking and pooling are deferred to
eval time so any dilation radius can be tested without re-running the backbone.

    python cache_features_to_npz.py \
        --task scs \
        --model-name raidium/curia \
        --batch-size 32

Each NPZ file gains a "patch_tokens" key of shape (N, hidden_size) float32.
Files that already have the "patch_tokens" key are skipped (idempotent).
Storage: ~3 MB per file (1024 patches × 768 dims × float32).
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, Dinov2Model

_DATA_ROOT = Path("/home/ge.polymtl.ca/p123239/data")
TASK_CONFIG = {
    "nfn": {"data_dir": _DATA_ROOT / "patches_RSNA_raw_with_mask_nfn"},
    "ss":  {"data_dir": _DATA_ROOT / "patches_RSNA_raw_with_mask_ss"},
    "scs": {"data_dir": _DATA_ROOT / "patches_RSNA_raw_with_mask_scs"},
}


def collect_npz_paths(data_dir: Path) -> list[Path]:
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
) -> dict[Path, np.ndarray]:
    """Run backbone on a batch; return {path: patch_tokens (N, D)}."""
    images = [np.load(p)["slice"].astype(np.float32) for p in paths]

    pv = processor(images, return_tensors="pt")["pixel_values"].to(device)
    outputs = backbone(pixel_values=pv, output_hidden_states=False)

    # Exclude CLS token (index 0) → patch tokens only
    patch_tokens = outputs.last_hidden_state[:, 1:].cpu().float().numpy()  # (B, N, D)
    return {p: patch_tokens[i] for i, p in enumerate(paths)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=["nfn", "ss", "scs"])
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--model-name", default="raidium/curia",
                    help="Path or HF repo of the curia backbone.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--force", action="store_true",
                    help="Recompute and overwrite existing 'patch_tokens' entries.")
    args = ap.parse_args()

    data_dir   = args.data_dir or TASK_CONFIG[args.task]["data_dir"]
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading backbone from {args.model_name} ...")
    backbone  = Dinov2Model.from_pretrained(args.model_name, trust_remote_code=True).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    N = (processor.crop_size // backbone.config.patch_size) ** 2
    D = backbone.config.hidden_size
    print(f"  patch_tokens shape per slice: ({N}, {D})  ~{N*D*4/1e6:.1f} MB/file  device={device}")

    all_paths = collect_npz_paths(data_dir)
    print(f"Found {len(all_paths)} NPZ files under {data_dir}")

    if args.force:
        todo = all_paths
        print(f"  --force: recomputing all {len(todo)} files.")
    else:
        todo = [p for p in all_paths if "patch_tokens" not in np.load(p).files]
        print(f"  {len(all_paths) - len(todo)} already cached, {len(todo)} to process.")

    if not todo:
        print("Nothing to do.")
        return

    n_saved = 0
    for i in tqdm(range(0, len(todo), args.batch_size), desc="Caching patch tokens"):
        batch_paths = todo[i: i + args.batch_size]
        token_map = process_batch(batch_paths, backbone, processor, device)

        for p, tokens in token_map.items():
            d = np.load(p)
            data = {k: d[k] for k in d.files}  # preserve slice, mask, …
            data["patch_tokens"] = tokens        # (N, D) float32
            np.savez_compressed(p, **data)
            n_saved += 1

    print(f"\nDone. patch_tokens cached in {n_saved}/{len(all_paths)} files.")


if __name__ == "__main__":
    main()
