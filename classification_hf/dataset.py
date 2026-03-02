from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def normalize_image_array(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-image z-score normalization — same as in segmentation_hf."""
    mean = arr.mean()
    std = arr.std()
    return (arr - mean) / (std + eps)


class ImageClassificationDataset(Dataset):
    """Loads raw images from class subdirectories for feature extraction.

    Expected layout:
        data_dir/
            class_a/  ← label 0
                img1.png
            class_b/  ← label 1
                img2.png
    Classes are sorted alphabetically; their indices become integer labels.
    """

    def __init__(self, data_dir: str, image_size: int = 512):
        self.image_size = image_size
        self.items: List[Tuple[Path, int]] = []

        data_path = Path(data_dir)
        class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
        if not class_dirs:
            raise ValueError(f"No subdirectories found in {data_dir}")

        self.class_names: List[str] = [d.name for d in class_dirs]
        self.num_classes = len(self.class_names)

        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        for label, class_dir in enumerate(class_dirs):
            files = sorted([f for f in class_dir.iterdir() if f.suffix.lower() in exts])
            for f in files:
                self.items.append((f, label))

        if not self.items:
            raise ValueError(f"No images found in {data_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.items[idx]
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32)
        arr_norm = normalize_image_array(arr / 255.0)
        t = torch.from_numpy(np.ascontiguousarray(arr_norm)).unsqueeze(0)  # (1, H, W)
        return t, label

    @property
    def class_counts(self) -> np.ndarray:
        labels = np.array([label for _, label in self.items])
        return np.bincount(labels, minlength=self.num_classes)

    @property
    def paths(self) -> List[str]:
        return [str(p) for p, _ in self.items]

    @property
    def labels(self) -> np.ndarray:
        return np.array([label for _, label in self.items])


class FeatureDataset(Dataset):
    """Loads pre-extracted features from a .npz file for fast classifier training.

    Expected .npz keys:
        features  : (N, D) float32
        labels    : (N,)   int64
        paths     : (N,)   str    [optional]
        class_names: (C,)  str    [optional]
    """

    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        self.features: np.ndarray = data["features"].astype(np.float32)
        self.labels: np.ndarray = data["labels"].astype(np.int64)
        self.file_paths: Optional[np.ndarray] = data.get("paths", None)

        if "class_names" in data:
            self.class_names: List[str] = list(data["class_names"])
        else:
            n = int(self.labels.max()) + 1
            self.class_names = [str(i) for i in range(n)]

        self.num_classes = len(self.class_names)
        self.feature_dim = self.features.shape[1]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return torch.from_numpy(self.features[idx]), int(self.labels[idx])

    @property
    def class_counts(self) -> np.ndarray:
        return np.bincount(self.labels, minlength=self.num_classes)
