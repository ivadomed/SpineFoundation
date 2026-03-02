"""
Extract frozen backbone features from an image classification dataset and save
them as .npz files for fast downstream training (no inference per epoch).

Usage:
    python -m classification_hf.extract_features \
        --model_dir /path/to/hf_checkpoint \
        --data_dir  /path/to/RSNA_patches_512 \
        --output_dir /path/to/features \
        --batch_size 64 --amp
"""

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ExtractConfig, parse_extract_args
from .dataset import ImageClassificationDataset
from .model import FrozenBackboneForExtraction


def extract_features(cfg: ExtractConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load backbone ──────────────────────────────────────────────────────────
    print(f"Loading backbone from: {cfg.model_dir}")
    model = FrozenBackboneForExtraction(cfg.model_dir).to(device)
    print(f"  hidden_size={model.hidden_size}  patch_size={model.patch_size}")

    # ── Load images ────────────────────────────────────────────────────────────
    dataset = ImageClassificationDataset(cfg.data_dir, image_size=cfg.image_size)
    print(f"\nDataset: {len(dataset)} images  —  {dataset.num_classes} classes")
    counts = dataset.class_counts
    for name, count in zip(dataset.class_names, counts):
        print(f"  class {name:>4s}: {count:>6d} images  ({100*count/len(dataset):.1f}%)")

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # ── Extract ────────────────────────────────────────────────────────────────
    amp_ctx = (
        torch.amp.autocast("cuda", enabled=True)
        if cfg.amp and device.type == "cuda"
        else nullcontext()
    )

    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extracting"):
            images = images.to(device, non_blocking=True)
            with amp_ctx:
                features = model(images)  # (B, hidden_size) — CLS token
            all_features.append(features.float().cpu().numpy())
            all_labels.append(labels.numpy())

    features_np = np.concatenate(all_features, axis=0)   # (N, D)
    labels_np   = np.concatenate(all_labels,   axis=0)   # (N,)
    paths_np    = np.array(dataset.paths)                 # (N,)
    class_names = np.array(dataset.class_names)

    print(f"\nExtracted features: {features_np.shape}  dtype={features_np.dtype}")

    # ── Stratified train/val split ─────────────────────────────────────────────
    indices = np.arange(len(features_np))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=cfg.val_split,
        random_state=cfg.seed,
        stratify=labels_np,
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, idx in [("train", train_idx), ("val", val_idx)]:
        out_path = output_dir / f"{split}.npz"
        np.savez(
            out_path,
            features=features_np[idx],
            labels=labels_np[idx],
            paths=paths_np[idx],
            class_names=class_names,
        )
        split_counts = np.bincount(labels_np[idx], minlength=len(class_names))
        print(f"\nSaved {split}: {len(idx)} samples → {out_path}")
        for name, cnt in zip(class_names, split_counts):
            print(f"  class {name:>4s}: {cnt:>6d}  ({100*cnt/len(idx):.1f}%)")

    print("\nDone.")


if __name__ == "__main__":
    cfg = parse_extract_args()
    extract_features(cfg)
