import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile, PngImagePlugin
from torch.utils.data import DataLoader, Dataset

from .config import TrainConfig

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
PngImagePlugin.MAX_TEXT_CHUNK = 16 * (1024**2)


def zscore_normalize(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = t.mean()
    std = t.std()
    return (t - mean) / (std + eps)


def pad_to_min_hw(arr: np.ndarray, min_h: int, min_w: int, fill: int = 0) -> np.ndarray:
    h, w = arr.shape[:2]
    pad_h = max(0, min_h - h)
    pad_w = max(0, min_w - w)
    if pad_h == 0 and pad_w == 0:
        return arr

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    if arr.ndim == 2:
        return np.pad(arr, ((top, bottom), (left, right)), mode="constant", constant_values=fill)
    return np.pad(arr, ((top, bottom), (left, right), (0, 0)), mode="constant", constant_values=fill)


def make_sliding_positions(full_size: int, tile_size: int, overlap: int) -> List[int]:
    stride = max(1, tile_size - overlap)
    if full_size <= tile_size:
        return [0]

    positions = list(range(0, full_size - tile_size + 1, stride))
    last = full_size - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def extract_tile_pair(
    image: np.ndarray,
    mask: np.ndarray,
    x0: int,
    y0: int,
    tile_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    image = pad_to_min_hw(image, tile_size, tile_size, fill=0)
    mask = pad_to_min_hw(mask, tile_size, tile_size, fill=0)
    image_tile = image[y0 : y0 + tile_size, x0 : x0 + tile_size]
    mask_tile = mask[y0 : y0 + tile_size, x0 : x0 + tile_size]
    return image_tile, mask_tile


def list_image_files(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


class PairedSegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_size: int,
        tile_overlap: int,
        split: str = "train",
        seed: int = 42,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.tile_overlap = tile_overlap
        self.split = split
        self.rng = random.Random(seed)

        if self.tile_overlap < 0 or self.tile_overlap >= self.image_size:
            raise ValueError("tile_overlap must satisfy 0 <= tile_overlap < image_size")

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        image_files = list_image_files(self.image_dir)
        if not image_files:
            raise ValueError(f"No images found in {self.image_dir}")

        pairs: List[Tuple[Path, Path]] = []
        missing = []
        for img_path in image_files:
            mask_path = self.mask_dir / img_path.name
            if not mask_path.exists():
                missing.append(img_path.name)
            else:
                pairs.append((img_path, mask_path))

        if missing:
            sample = ", ".join(missing[:5])
            raise ValueError(
                f"Missing masks for {len(missing)} image(s) in {self.mask_dir}. Examples: {sample}"
            )

        self.pairs = pairs
        self.tiles: List[Tuple[int, int, int]] = []

        for pair_idx, (img_path, _) in enumerate(self.pairs):
            img = Image.open(img_path).convert("L")
            arr = np.array(img, dtype=np.float32)
            arr = pad_to_min_hw(arr, self.image_size, self.image_size, fill=0)
            h, w = arr.shape[:2]

            xs = make_sliding_positions(w, self.image_size, self.tile_overlap)
            ys = make_sliding_positions(h, self.image_size, self.tile_overlap)

            for y0 in ys:
                for x0 in xs:
                    self.tiles.append((pair_idx, x0, y0))

    def __len__(self) -> int:
        return len(self.tiles)

    def _load_image_grayscale(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("L")
        return np.array(img, dtype=np.float32)

    def _load_mask(self, path: Path) -> np.ndarray:
        mask = Image.open(path).convert("L")
        mask_np = np.array(mask, dtype=np.float32)
        mask_np = (mask_np > 0).astype(np.float32)
        return mask_np

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pair_idx, x0, y0 = self.tiles[idx]
        img_path, mask_path = self.pairs[pair_idx]
        image = self._load_image_grayscale(img_path)
        mask = self._load_mask(mask_path)

        image, mask = extract_tile_pair(image, mask, x0=x0, y0=y0, tile_size=self.image_size)

        image_t = torch.from_numpy(image).unsqueeze(0) / 255.0
        image_t = zscore_normalize(image_t)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        return image_t, mask_t


def build_dataloaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    train_ds = PairedSegmentationDataset(
        image_dir=cfg.train_images,
        mask_dir=cfg.train_masks,
        image_size=cfg.image_size,
        tile_overlap=cfg.tile_overlap,
        split="train",
        seed=cfg.seed,
    )
    val_ds = PairedSegmentationDataset(
        image_dir=cfg.val_images,
        mask_dir=cfg.val_masks,
        image_size=cfg.image_size,
        tile_overlap=cfg.tile_overlap,
        split="val",
        seed=cfg.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader
