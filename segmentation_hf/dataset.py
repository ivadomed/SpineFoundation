from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile, PngImagePlugin
from torch.utils.data import DataLoader, Dataset

from .config import TrainConfig

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
PngImagePlugin.MAX_TEXT_CHUNK = 16 * (1024**2)


@dataclass
class TileRecord:
    pair_idx: int
    x0: int
    y0: int
    must_tile: bool


def zscore_normalize(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = t.mean()
    std = t.std()
    return (t - mean) / (std + eps)


def overlap_pct_to_pixels(tile_size: int, overlap_pct: float) -> int:
    if overlap_pct < 0.0 or overlap_pct >= 100.0:
        raise ValueError("tile_overlap_pct must satisfy 0 <= pct < 100")
    overlap_px = int(round(tile_size * (overlap_pct / 100.0)))
    overlap_px = max(0, min(tile_size - 1, overlap_px))
    return overlap_px


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


def make_sliding_positions(full_size: int, tile_size: int, overlap_px: int) -> List[int]:
    stride = max(1, tile_size - overlap_px)
    if full_size <= tile_size:
        return [0]

    positions = list(range(0, full_size - tile_size + 1, stride))
    last = full_size - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def list_image_files(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


class PairedSegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_size: int,
        tile_overlap_pct: float,
        tile_threshold: int,
        split: str = "train",
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.tile_overlap_pct = tile_overlap_pct
        self.tile_threshold = tile_threshold
        self.split = split
        self.tile_overlap_px = overlap_pct_to_pixels(tile_size=image_size, overlap_pct=tile_overlap_pct)

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
        self.tiles: List[TileRecord] = []

        for pair_idx, (img_path, _) in enumerate(self.pairs):
            with Image.open(img_path) as img:
                arr = np.array(img.convert("L"), dtype=np.uint8)

            h, w = arr.shape[:2]
            must_tile = max(h, w) > self.tile_threshold
            if not must_tile:
                self.tiles.append(TileRecord(pair_idx=pair_idx, x0=0, y0=0, must_tile=False))
                continue

            arr_pad = pad_to_min_hw(arr, self.image_size, self.image_size, fill=0)
            hp, wp = arr_pad.shape[:2]
            xs = make_sliding_positions(wp, self.image_size, self.tile_overlap_px)
            ys = make_sliding_positions(hp, self.image_size, self.tile_overlap_px)

            for y0 in ys:
                for x0 in xs:
                    self.tiles.append(TileRecord(pair_idx=pair_idx, x0=x0, y0=y0, must_tile=True))

    def __len__(self) -> int:
        return len(self.tiles)

    def _load_image_grayscale(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("L")
        return np.array(img, dtype=np.float32)

    def _load_mask(self, path: Path) -> np.ndarray:
        mask = Image.open(path).convert("L")
        mask_np = np.array(mask, dtype=np.float32)
        return (mask_np > 0).astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self.tiles[idx]
        img_path, mask_path = self.pairs[rec.pair_idx]
        image = self._load_image_grayscale(img_path)
        mask = self._load_mask(mask_path)

        if rec.must_tile:
            image = pad_to_min_hw(image, self.image_size, self.image_size, fill=0)
            mask = pad_to_min_hw(mask, self.image_size, self.image_size, fill=0)
            image = image[rec.y0 : rec.y0 + self.image_size, rec.x0 : rec.x0 + self.image_size]
            mask = mask[rec.y0 : rec.y0 + self.image_size, rec.x0 : rec.x0 + self.image_size]

        image_t = torch.from_numpy(image).unsqueeze(0) / 255.0
        image_t = zscore_normalize(image_t)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        return image_t, mask_t


def collate_pad_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_to_multiple: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    images = [b[0] for b in batch]
    masks = [b[1] for b in batch]

    max_h = max(t.shape[-2] for t in images)
    max_w = max(t.shape[-1] for t in images)

    if pad_to_multiple > 1:
        max_h = ((max_h + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
        max_w = ((max_w + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple

    padded_images = []
    padded_masks = []
    for image, mask in zip(images, masks):
        pad_h = max_h - image.shape[-2]
        pad_w = max_w - image.shape[-1]
        padded_images.append(F.pad(image, (0, pad_w, 0, pad_h), mode="constant", value=0.0))
        padded_masks.append(F.pad(mask, (0, pad_w, 0, pad_h), mode="constant", value=0.0))

    return torch.stack(padded_images, dim=0), torch.stack(padded_masks, dim=0)


def build_datasets(cfg: TrainConfig) -> Tuple[PairedSegmentationDataset, PairedSegmentationDataset]:
    train_ds = PairedSegmentationDataset(
        image_dir=cfg.train_images,
        mask_dir=cfg.train_masks,
        image_size=cfg.image_size,
        tile_overlap_pct=cfg.tile_overlap_pct,
        tile_threshold=cfg.tile_threshold,
        split="train",
    )
    val_ds = PairedSegmentationDataset(
        image_dir=cfg.val_images,
        mask_dir=cfg.val_masks,
        image_size=cfg.image_size,
        tile_overlap_pct=cfg.tile_overlap_pct,
        tile_threshold=cfg.tile_threshold,
        split="val",
    )
    return train_ds, val_ds


def build_train_dataloader(cfg: TrainConfig, train_ds: PairedSegmentationDataset) -> DataLoader:
    return DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_pad_batch,
    )
