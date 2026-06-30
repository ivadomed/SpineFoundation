import hashlib
import pickle
import random
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
    """Per-tensor z-score normalization (kept for backward compatibility)."""
    mean = t.mean()
    std = t.std()
    return (t - mean) / (std + eps)


def normalize_image_array(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-image z-score normalization on a float32 array (values in [0, 1]).

    Normalizing the full image before tiling prevents unstable statistics on
    nearly-uniform tiles (e.g. background-only tiles whose std ≈ 0).
    """
    mean = arr.mean()
    std = arr.std()
    return (arr - mean) / (std + eps)


def overlap_pct_to_pixels(tile_size: int, overlap_pct: float) -> int:
    if overlap_pct < 0.0 or overlap_pct >= 100.0:
        raise ValueError("tile_overlap_pct must satisfy 0 <= pct < 100")
    overlap_px = int(round(tile_size * (overlap_pct / 100.0)))
    overlap_px = max(0, min(tile_size - 1, overlap_px))
    return overlap_px


def pad_to_min_hw(arr: np.ndarray, min_h: int, min_w: int, fill: float = 0.0) -> np.ndarray:
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


def list_image_files(folder: Path, only_sagittal: bool = False, only_axial: bool = False) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    if only_sagittal and only_axial:
        raise ValueError("only_sagittal and only_axial cannot both be enabled")
    files = []
    for p in folder.rglob("*"):
        if not (p.is_file() and p.suffix.lower() in exts):
            continue
        name_lower = p.name.lower()
        if only_sagittal and "sag" not in name_lower:
            continue
        if only_axial and "ax" not in name_lower:
            continue
        files.append(p)
    return sorted(files)


class PairedSegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_size: int,
        only_sagittal: bool,
        only_axial: bool,
        tile_overlap_pct: float,
        tile_threshold: int,
        split: str = "train",
        augment: bool = False,
        cache_dir: str | None = None,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.only_sagittal = only_sagittal
        self.only_axial = only_axial
        self.tile_overlap_pct = tile_overlap_pct
        self.tile_threshold = tile_threshold
        self.split = split
        self.augment = augment
        self.tile_overlap_px = overlap_pct_to_pixels(tile_size=image_size, overlap_pct=tile_overlap_pct)

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        # ── index cache ──────────────────────────────────────────────────────
        cache_key = hashlib.md5(
            f"{self.image_dir}|{self.mask_dir}|{image_size}|{only_sagittal}|{only_axial}|{tile_threshold}|{tile_overlap_pct}".encode()
        ).hexdigest()
        cache_path = Path(cache_dir) / f"index_{cache_key}.pkl" if cache_dir else None

        if cache_path and cache_path.exists():
            print(f"[dataset:{split}] loading index from cache ({cache_path.name})")
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            self.pairs: List[Tuple[str, str]] = cached["pairs"]
            self.tiles: List[TileRecord] = cached["tiles"]
            print(f"[dataset:{split}] {len(self.pairs)} pairs, {len(self.tiles)} tiles (from cache)")
        else:
            image_files = list_image_files(self.image_dir, only_sagittal=self.only_sagittal, only_axial=self.only_axial)
            if not image_files:
                mode = "sagittal-only" if self.only_sagittal else ("axial-only" if self.only_axial else "all-planes")
                raise ValueError(f"No images found in {self.image_dir} (mode={mode})")

            pairs_raw: List[Tuple[str, str]] = []
            n_missing = 0
            for img_path in image_files:
                rel = img_path.relative_to(self.image_dir)
                mask_path = self.mask_dir / rel
                if not mask_path.exists():
                    n_missing += 1
                else:
                    pairs_raw.append((str(img_path), str(mask_path)))

            if n_missing > 0:
                print(f"[dataset:{split}] {n_missing}/{len(image_files)} images have no matching mask — skipped.")

            self.pairs = pairs_raw
            # One entry per pair — no tiling, images are resized on-the-fly in __getitem__
            self.tiles = list(range(len(self.pairs)))

            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump({"pairs": self.pairs, "tiles": self.tiles}, f)
                print(f"[dataset:{split}] index cached to {cache_path.name}")

            print(f"[dataset:{split}] {len(self.pairs)} pairs")

        # Per-worker lazy cache: populated on first access within each worker process.
        # Avoids reloading the same image from disk N times per epoch when it produces N tiles.
        self._img_cache: dict[int, np.ndarray] = {}
        self._mask_cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.tiles)

    def _load_image_grayscale(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("L")
        return np.array(img, dtype=np.float32)

    def _load_mask(self, path: Path) -> np.ndarray:
        mask = Image.open(path).convert("L")
        mask_np = np.array(mask, dtype=np.float32)
        return (mask_np > 0).astype(np.float32)

    def _get_cached_image(self, pair_idx: int, path: Path) -> np.ndarray:
        if pair_idx not in self._img_cache:
            self._img_cache[pair_idx] = self._load_image_grayscale(path)
        return self._img_cache[pair_idx]

    def _get_cached_mask(self, pair_idx: int, path: Path) -> np.ndarray:
        if pair_idx not in self._mask_cache:
            self._mask_cache[pair_idx] = self._load_mask(path)
        return self._mask_cache[pair_idx]

    def load_raw_pair(self, pair_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (image_float32_0_255, mask_float32_binary) for a given pair index."""
        img_path, mask_path = self.pairs[pair_idx]
        return self._get_cached_image(pair_idx, img_path), self._get_cached_mask(pair_idx, mask_path)

    @staticmethod
    def _augment(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Intensity-only augmentations. No spatial transforms: spine MRI has fixed orientation."""
        # Additive Gaussian noise
        if random.random() < 0.5:
            image = image + np.random.normal(0.0, 0.05, image.shape).astype(np.float32)
        # Random brightness/contrast shift (image only)
        if random.random() < 0.5:
            image = image * random.uniform(0.85, 1.15) + random.uniform(-0.1, 0.1)
        return image, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.pairs[self.tiles[idx]]

        with Image.open(img_path) as img:
            img_resized = img.convert("L").resize((self.image_size, self.image_size), Image.BILINEAR)
        with Image.open(mask_path) as msk:
            msk_resized = msk.convert("L").resize((self.image_size, self.image_size), Image.NEAREST)

        image = np.array(img_resized, dtype=np.float32)
        mask = (np.array(msk_resized, dtype=np.float32) > 0).astype(np.float32)

        image_norm = normalize_image_array(image / 255.0)

        if self.augment:
            image_norm, mask = self._augment(image_norm, mask)

        image_t = torch.from_numpy(np.ascontiguousarray(image_norm)).unsqueeze(0)
        mask_t = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)
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
        only_sagittal=cfg.only_sagittal,
        only_axial=cfg.only_axial,
        tile_overlap_pct=cfg.tile_overlap_pct,
        tile_threshold=cfg.tile_threshold,
        split="train",
        augment=cfg.augment,
        cache_dir=cfg.cache_dir,
    )
    val_ds = PairedSegmentationDataset(
        image_dir=cfg.val_images,
        mask_dir=cfg.val_masks,
        image_size=cfg.image_size,
        only_sagittal=cfg.only_sagittal,
        only_axial=cfg.only_axial,
        tile_overlap_pct=cfg.tile_overlap_pct,
        tile_threshold=cfg.tile_threshold,
        split="val",
        augment=False,
        cache_dir=cfg.cache_dir,
    )
    return train_ds, val_ds


def build_train_dataloader(cfg: TrainConfig, train_ds: PairedSegmentationDataset) -> DataLoader:
    from torch.utils.data import WeightedRandomSampler
    import numpy as np
    from PIL import Image

    # Compute per-sample weights: oversample images that contain foreground (spine)
    weights = []
    for img_path, mask_path in train_ds.pairs:
        mask = np.array(Image.open(mask_path).convert("L"))
        has_fg = float(mask.max() > 0)
        # Positive images get weight 2, negatives get weight 1
        weights.append(1.0 + has_fg)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    return DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_pad_batch,
    )


def build_val_dataloader(cfg: TrainConfig, val_ds: PairedSegmentationDataset) -> DataLoader:
    return DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_pad_batch,
    )


# ── NPZ-backed dataset (pre-cached patch tokens, fast path) ──────────────────

def list_npz_files(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".npz"])


class NpzSegmentationDataset(Dataset):
    """Segmentation dataset backed by NPZ files.

    Each NPZ is expected to contain:
      - "slice": float32 (H, W)  — grayscale intensity slice
      - "mask":  uint8 or float32 (H, W) — binary annotation mask

    After running cache_patch_tokens.py, each NPZ will also contain:
      - token_key: float32 (N, D) — pre-cached backbone patch tokens (fast path)

    Fast path (tokens present):
        __getitem__ returns (patch_tokens: Tensor[N, D], mask: Tensor[1, S, S])
        where S = image_size (backbone input resolution).

    Slow path (no tokens):
        __getitem__ returns (image: Tensor[1, S, S], mask: Tensor[1, S, S])
        normalized and cropped/padded to image_size.
    """

    def __init__(
        self,
        data_dir: str,
        image_size: int,
        token_key: str = "patch_tokens",
        augment: bool = False,
        preload: bool = True,
        max_preload_samples: int = 2000,
        verbose: bool = True,
    ):
        self.data_dir   = Path(data_dir)
        self.image_size = image_size
        self.token_key  = token_key
        self.augment    = augment

        if not self.data_dir.exists():
            raise FileNotFoundError(f"NPZ directory not found: {self.data_dir}")

        self.npz_paths = list_npz_files(self.data_dir)
        if not self.npz_paths:
            raise ValueError(f"No NPZ files found in {self.data_dir}")

        # Inspect the first file to detect fast-path availability.
        d = np.load(self.npz_paths[0])
        self.has_tokens = token_key in d.files

        # Expose the same attributes as PairedSegmentationDataset for eval
        # compatibility (predict_full_image_detiled uses tile_overlap_pct /
        # tile_threshold when the slow backbone path is needed for overlays).
        self.tile_overlap_pct = 0.0
        self.tile_threshold   = self.image_size + 1  # never tile NPZ images

        # pairs attribute: list of (img_path, mask_path) expected by
        # capture_val_overlays — we expose npz_paths under both attributes.
        self.pairs = [(p, p) for p in self.npz_paths]

        # Preload all tokens + masks into RAM, using /dev/shm as a cross-process
        # cache so that subsequent trials (same data_dir + token_key) skip the
        # NPZ loading entirely and read straight from shared RAM.
        self._tokens_cache: list | None = None
        self._masks_cache:  list | None = None
        if preload and self.has_tokens and len(self.npz_paths) <= max_preload_samples:
            import hashlib, os
            cache_key  = hashlib.md5(f"{self.data_dir}|{token_key}|{image_size}".encode()).hexdigest()[:12]
            cache_dir  = Path(os.environ.get("NPZ_CACHE_DIR", Path.home() / ".cache" / "npz_tokens"))
            cache_dir.mkdir(parents=True, exist_ok=True)
            shm_path   = cache_dir / f"npz_cache_{cache_key}.pt"

            if shm_path.exists():
                if verbose: print(f"[NpzDataset] Loading cache from {shm_path} …", flush=True)
                cached = torch.load(shm_path, map_location="cpu", weights_only=False)
                self._tokens_cache = cached["tokens"]
                self._masks_cache  = cached["masks"]
                if verbose: print(f"[NpzDataset] Cache loaded ({len(self._tokens_cache)} samples).", flush=True)
            else:
                if verbose: print(f"[NpzDataset] Preloading {len(self.npz_paths)} samples into RAM …", flush=True)
                tokens_list, masks_list = [], []
                for p in self.npz_paths:
                    npz = np.load(p)
                    tokens_list.append(torch.from_numpy(npz[token_key].astype(np.float32)))
                    mask = (npz["mask"].astype(np.float32) > 0).astype(np.float32)
                    mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
                    mask_t = F.interpolate(
                        mask_t, size=(image_size, image_size), mode="nearest"
                    ).squeeze(0)
                    masks_list.append(mask_t)
                self._tokens_cache = tokens_list
                self._masks_cache  = masks_list
                if verbose: print(f"[NpzDataset] Saving cache to {shm_path} …", flush=True)
                torch.save({"tokens": tokens_list, "masks": masks_list}, shm_path)
                if verbose: print(f"[NpzDataset] Cache saved ({shm_path.name}).", flush=True)

    def __len__(self) -> int:
        return len(self.npz_paths)

    def load_raw_pair(self, pair_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (image_float32, mask_float32_binary) for full-image eval.

        Compatible with PairedSegmentationDataset.load_raw_pair() so that
        predict_full_image_detiled() works unchanged for backbone-based eval
        (e.g. W&B overlays).
        """
        d = np.load(self.npz_paths[pair_idx])
        image = d["slice"].astype(np.float32)
        mask  = (d["mask"].astype(np.float32) > 0).astype(np.float32)
        return image, mask

    def __getitem__(self, idx: int):
        if self.has_tokens:
            # Fast path: serve from RAM cache if available.
            if self._tokens_cache is not None:
                return self._tokens_cache[idx], self._masks_cache[idx]
            d = np.load(self.npz_paths[idx])
            patch_tokens = torch.from_numpy(d[self.token_key].astype(np.float32))  # (N, D)
            mask = (d["mask"].astype(np.float32) > 0).astype(np.float32)           # (H, W)
            mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)              # (1, 1, H, W)
            mask_t = F.interpolate(
                mask_t, size=(self.image_size, self.image_size), mode="nearest"
            ).squeeze(0)                                                            # (1, S, S)
            return patch_tokens, mask_t

        # Slow path: normalize image and return for backbone inference.
        image = d["slice"].astype(np.float32)
        mask  = (d["mask"].astype(np.float32) > 0).astype(np.float32)
        image_norm = normalize_image_array(image / 255.0 if image.max() > 1.0 else image)
        image_pad  = pad_to_min_hw(image_norm, self.image_size, self.image_size, fill=0.0)
        mask_pad   = pad_to_min_hw(mask, self.image_size, self.image_size, fill=0.0)
        image_crop = image_pad[:self.image_size, :self.image_size]
        mask_crop  = mask_pad[:self.image_size, :self.image_size]

        if self.augment:
            image_crop, mask_crop = PairedSegmentationDataset._augment(image_crop, mask_crop)

        image_t = torch.from_numpy(np.ascontiguousarray(image_crop)).unsqueeze(0)
        mask_t  = torch.from_numpy(np.ascontiguousarray(mask_crop)).unsqueeze(0)
        return image_t, mask_t


def build_npz_datasets(cfg: "TrainConfig", verbose: bool = True) -> Tuple[NpzSegmentationDataset, NpzSegmentationDataset]:
    """Build train/val NPZ-backed datasets from cfg.npz_train_dir / cfg.npz_val_dir."""
    train_ds = NpzSegmentationDataset(
        data_dir=cfg.npz_train_dir,
        image_size=cfg.image_size,
        token_key=cfg.patch_token_key,
        augment=cfg.augment,
        max_preload_samples=cfg.max_preload_samples,
        verbose=verbose,
    )
    val_ds = NpzSegmentationDataset(
        data_dir=cfg.npz_val_dir,
        image_size=cfg.image_size,
        token_key=cfg.patch_token_key,
        augment=False,
        max_preload_samples=cfg.max_preload_samples,
        verbose=verbose,
    )
    return train_ds, val_ds


def build_npz_train_dataloader(cfg: "TrainConfig", train_ds: NpzSegmentationDataset) -> DataLoader:
    preloaded = train_ds._tokens_cache is not None
    nw = 0  # workers fork parent RAM → memory pressure; disk reads are fast enough without workers
    return DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=not preloaded,
        drop_last=False,
        persistent_workers=(nw > 0),
        prefetch_factor=(4 if nw > 0 else None),
    )


def build_npz_val_dataloader(cfg: "TrainConfig", val_ds: NpzSegmentationDataset) -> DataLoader:
    preloaded = val_ds._tokens_cache is not None
    nw = 0  # workers fork parent RAM → memory pressure; disk reads are fast enough without workers
    return DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=not preloaded,
        drop_last=False,
        persistent_workers=(nw > 0),
        prefetch_factor=(4 if nw > 0 else None),
    )
