"""
Local-directory dataset builder that mirrors curia's pipeline.

Loads images from a folder tree (one sub-dir per class), converts them to a
HuggingFace Dataset, then applies the same AutoImageProcessor used by curia.

Expected layout:
    data_dir/
        0/   <- label 0
            img1.png
        1/   <- label 1
            img2.png
Classes are sorted alphabetically; their index becomes the integer label.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# ── Build a HuggingFace DatasetDict from a local directory ────────────────────

def load_local_dataset(data_dir: str, val_split: float = 0.15, seed: int = 42) -> DatasetDict:
    """
    Returns a DatasetDict with "train" and "val" splits built from a local
    directory tree (one sub-dir per class).  Attaches .class_names attribute.
    """
    data_path = Path(data_dir)
    class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No sub-directories found in {data_dir}")

    class_names: List[str] = [d.name for d in class_dirs]
    paths: List[str] = []
    labels: List[int] = []

    for label, class_dir in enumerate(class_dirs):
        files = sorted([f for f in class_dir.iterdir() if f.suffix.lower() in _IMG_EXTS])
        for f in files:
            paths.append(str(f))
            labels.append(label)

    if not paths:
        raise ValueError(f"No images found in {data_dir}")

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
    Load images from disk, run through AutoImageProcessor (bicubic resize +
    per-image z-score), return pixel_values + labels — identical to curia.
    """
    images_as_np = []
    for path in examples["path"]:
        img = Image.open(path)
        if img.mode not in ("L", "F"):
            img = img.convert("L")
        images_as_np.append(np.array(img, dtype=np.float32))

    processed = processor(images_as_np, return_tensors="pt")
    return {
        "pixel_values": processed["pixel_values"],
        "labels": torch.tensor(examples["target"], dtype=torch.long),
    }


# ── Feature extraction: mirror curia's extract_features (2-D, no mask) ────────

def extract_features_fn(examples: Dict, processor, backbone) -> Dict:
    """
    Run frozen backbone on a batch, store CLS token as pixel_values.
    The key name 'pixel_values' is intentional: Classifier.forward expects it.
    Mirrors curia's extract_features() for 2-D images without masks.
    """
    images_as_np = []
    for path in examples["path"]:
        img = Image.open(path)
        if img.mode not in ("L", "F"):
            img = img.convert("L")
        images_as_np.append(np.array(img, dtype=np.float32))

    processed = processor(images_as_np, return_tensors="pt")
    pixel_values = processed["pixel_values"].cuda()

    with torch.no_grad():
        outputs = backbone(pixel_values=pixel_values, output_hidden_states=False)

    cls_tokens = outputs.last_hidden_state[:, 0].cpu()  # (B, hidden_size)

    return {
        "pixel_values": cls_tokens,
        "labels": torch.tensor(examples["target"], dtype=torch.long),
    }
