"""
Local-directory dataset builder that mirrors curia's pipeline.

Loads images from a folder tree (one sub-dir per class), converts them to a
HuggingFace Dataset, then applies the same AutoImageProcessor used by curia.

Expected layout:
    data_dir/
        0/   <- label 0
            img1.png   OR   img1.npz
        1/   <- label 1
            img2.png   OR   img2.npz
Classes are sorted alphabetically; their index becomes the integer label.

NPZ files must contain:
    slice : float32 (H, W)  — raw MRI intensity
    mask  : uint8   (H, W)  — binary annotation mask (optional but recommended)
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
_NPZ_EXT  = ".npz"

# DINOv2 patch size and dilation radius (must match eval_pretrained_nfn.py)
_DINO_PATCH = 14
_DILATE_R   = _DINO_PATCH // 2 + 1   # = 8


def resize_mask(mask_np: np.ndarray, target_size: int) -> torch.Tensor:
    """
    Map a sparse binary mask from its original resolution to target_size×target_size
    using explicit coordinate mapping (avoids nearest-neighbour pixel dropout for
    large images) followed by morphological dilation to guarantee coverage of the
    DINOv2 patch grid even in the 8-pixel dead zone at image edges.

    Returns: (1, 1, target_size, target_size) float32 tensor.
    """
    H, W = mask_np.shape
    t = torch.zeros(1, 1, target_size, target_size)
    ys, xs = np.where(mask_np > 0)
    for y, x in zip(ys, xs):
        y_out = min(int(y * target_size / H), target_size - 1)
        x_out = min(int(x * target_size / W), target_size - 1)
        t[0, 0, y_out, x_out] = 1.0
    t = F.max_pool2d(t, kernel_size=2 * _DILATE_R + 1, stride=1, padding=_DILATE_R)
    return t  # (1, 1, target_size, target_size)


def _masked_avg_pool(
    last_hidden_state: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Replicate extract_mask_features(use_avgpool=True) from modeling_dinov2.py.

    last_hidden_state : (B, 1+N, hidden_size)  — CLS token is index 0
    mask              : (B, 1, crop_size, crop_size)  — already resized + dilated

    Returns (B, hidden_size) masked-average-pooled patch features.
    """
    patch_tokens = last_hidden_state[:, 1:, :]          # skip CLS → (B, N, D)
    B, N, D = patch_tokens.shape
    grid = int(N ** 0.5)                                # 36 for 512/14

    # Downsample mask to patch grid exactly as the model does internally
    mask_pooled = F.max_pool2d(
        mask.float(), kernel_size=_DINO_PATCH, stride=_DINO_PATCH
    )                                                   # (B, 1, grid, grid)
    mask_flat = mask_pooled.view(B, grid * grid)        # (B, N)

    weights = mask_flat.unsqueeze(-1)                   # (B, N, 1)
    total   = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)  # (B, 1, 1)
    features = (patch_tokens * weights).sum(dim=1) / total.squeeze(1)  # (B, D)
    return features


# ── Build a HuggingFace DatasetDict from a local directory ────────────────────

def load_local_dataset(data_dir: str, val_split: float = 0.15, seed: int = 42) -> DatasetDict:
    """
    Returns a DatasetDict with "train" and "val" splits built from a local
    directory tree (one sub-dir per class).  Attaches .class_names attribute.
    Supports both image files (.png, .jpg, …) and .npz files.
    """
    data_path = Path(data_dir)
    class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No sub-directories found in {data_dir}")

    class_names: List[str] = [d.name for d in class_dirs]
    paths: List[str] = []
    labels: List[int] = []

    for label, class_dir in enumerate(class_dirs):
        files = sorted([
            f for f in class_dir.iterdir()
            if f.suffix.lower() in _IMG_EXTS or f.suffix.lower() == _NPZ_EXT
        ])
        for f in files:
            paths.append(str(f))
            labels.append(label)

    if not paths:
        raise ValueError(f"No images or NPZ files found in {data_dir}")

    labels_np = np.array(labels)
    indices   = np.arange(len(paths))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_split, random_state=seed, stratify=labels_np
    )

    def _make_split(idx):
        return Dataset.from_dict(
            {"path": [paths[i] for i in idx], "target": [labels[i] for i in idx]}
        )

    ds = DatasetDict({"train": _make_split(train_idx), "val": _make_split(val_idx)})
    ds.class_names = class_names  # type: ignore[attr-defined]
    return ds


# ── Preprocessing: mirror curia's preprocess_function ─────────────────────────

def preprocess_function(examples: Dict, processor) -> Dict:
    """
    Load images (PNG or NPZ) from disk, run through AutoImageProcessor (bicubic
    resize + per-image z-score), return pixel_values + labels (+ mask if NPZ).

    For NPZ files the binary mask is resized and dilated to match the processor's
    crop_size, then returned as a (1, crop_size, crop_size) tensor per sample so
    the HF Trainer can pass it directly to model(pixel_values=…, mask=…).
    """
    images_as_np: list = []
    masks: list = []

    for path in examples["path"]:
        p = Path(path)
        if p.suffix.lower() == _NPZ_EXT:
            d = np.load(path)
            images_as_np.append(d["slice"].astype(np.float32))
            if "mask" in d:
                crop_size = processor.crop_size
                mask_t = resize_mask(d["mask"], crop_size)  # (1, 1, S, S)
                masks.append(mask_t.squeeze(0))              # (1, S, S)
            else:
                masks.append(None)
        else:
            img = Image.open(path)
            if img.mode not in ("L", "F"):
                img = img.convert("L")
            images_as_np.append(np.array(img, dtype=np.float32))
            masks.append(None)

    processed = processor(images_as_np, return_tensors="pt")
    result = {
        "pixel_values": processed["pixel_values"],
        "labels": torch.tensor(examples["target"], dtype=torch.long),
    }

    if all(m is not None for m in masks):
        result["mask"] = torch.stack(masks, dim=0)  # (B, 1, S, S)

    return result


# ── Feature extraction: mirror curia's extract_features (2-D, with mask) ──────

def extract_features_fn(examples: Dict, processor, backbone) -> Dict:
    """
    Return backbone features for a batch of samples as "pixel_values".

    Fast path — pre-cached features in NPZ:
        If every path is an NPZ file that already contains a "features" key
        (written by cache_features_to_npz.py), the features are loaded directly
        without running the backbone.  backbone may be None in this case.

    Slow path — run backbone:
        For PNG files or NPZ without "features", the frozen backbone is run,
        then masked average pooling (or CLS fallback) is applied.

    The key name "pixel_values" is intentional: Classifier.forward expects it.
    """
    # ── Fast path: all features already cached in NPZ ─────────────────────────
    cached: list = []
    all_cached = True
    for path in examples["path"]:
        p = Path(path)
        if p.suffix.lower() == _NPZ_EXT:
            d = np.load(path)
            if "features" in d:
                cached.append(torch.from_numpy(d["features"].copy()))
                continue
        all_cached = False
        cached.append(None)

    if all_cached:
        return {
            "pixel_values": torch.stack(cached),
            "labels": torch.tensor(examples["target"], dtype=torch.long),
        }

    # ── Slow path: run backbone ────────────────────────────────────────────────
    images_as_np: list = []
    mask_tensors: list = []

    for path in examples["path"]:
        p = Path(path)
        if p.suffix.lower() == _NPZ_EXT:
            d = np.load(path)
            images_as_np.append(d["slice"].astype(np.float32))
            if "mask" in d:
                crop_size = processor.crop_size
                mask_tensors.append(resize_mask(d["mask"], crop_size))  # (1, 1, S, S)
            else:
                mask_tensors.append(None)
        else:
            img = Image.open(path)
            if img.mode not in ("L", "F"):
                img = img.convert("L")
            images_as_np.append(np.array(img, dtype=np.float32))
            mask_tensors.append(None)

    processed = processor(images_as_np, return_tensors="pt")
    pixel_values = processed["pixel_values"].cuda()

    with torch.no_grad():
        outputs = backbone(pixel_values=pixel_values, output_hidden_states=False)

    if all(m is not None for m in mask_tensors):
        mask_batch = torch.cat(mask_tensors, dim=0).cuda()  # (B, 1, S, S)
        features = _masked_avg_pool(outputs.last_hidden_state, mask_batch).cpu()
    else:
        features = outputs.last_hidden_state[:, 0].cpu()    # CLS token fallback

    return {
        "pixel_values": features,
        "labels": torch.tensor(examples["target"], dtype=torch.long),
    }
