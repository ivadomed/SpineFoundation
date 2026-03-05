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
    slice        : float32 (H, W)  — raw MRI intensity
    mask         : uint8   (H, W)  — binary annotation mask (optional but recommended)
    patch_tokens : float32 (N, D)  — pre-cached DINOv2 patch tokens (optional fast path)
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from datasets import Dataset, DatasetDict
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
_NPZ_EXT  = ".npz"

# DINOv2 patch size (fixed architecture property)
_DINO_PATCH = 14


def _get_crop_size(processor) -> int:
    """Handle both int and dict forms of processor.crop_size."""
    cs = processor.crop_size
    if isinstance(cs, dict):
        return cs["height"]
    return int(cs)


def resize_mask(mask_np: np.ndarray, target_size: int, dilation_radius: int = 8) -> torch.Tensor:
    """
    Map a sparse binary mask from its original resolution to target_size×target_size
    using vectorised coordinate mapping followed by morphological dilation.

    Returns: (1, 1, target_size, target_size) float32 tensor.
    """
    H, W = mask_np.shape
    ys, xs = np.where(mask_np > 0)
    t = torch.zeros(1, 1, target_size, target_size)
    if len(ys) > 0:
        ys_out = np.clip((ys * target_size / H).astype(np.int32), 0, target_size - 1)
        xs_out = np.clip((xs * target_size / W).astype(np.int32), 0, target_size - 1)
        t[0, 0, ys_out, xs_out] = 1.0
    if dilation_radius > 0:
        t = F.max_pool2d(t, kernel_size=2 * dilation_radius + 1, stride=1, padding=dilation_radius)
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
    return _masked_avg_pool_tokens(patch_tokens, mask)


def _masked_avg_pool_tokens(
    tokens: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Masked average pooling directly on patch tokens (no CLS token present).

    tokens : (B, N, hidden_size)
    mask   : (B, 1, crop_size, crop_size)  — already resized + dilated

    Returns (B, hidden_size).
    """
    B, N, D = tokens.shape
    mask_pooled = F.max_pool2d(
        mask.float(), kernel_size=_DINO_PATCH, stride=_DINO_PATCH
    )                                                   # (B, 1, grid, grid)
    mask_flat = mask_pooled.view(B, N)                  # (B, N)

    weights = mask_flat.unsqueeze(-1)                   # (B, N, 1)
    total   = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)  # (B, 1, 1)
    features = (tokens * weights).sum(dim=1) / total.squeeze(1)  # (B, D)
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

    crop_size = _get_crop_size(processor)

    for path in examples["path"]:
        p = Path(path)
        if p.suffix.lower() == _NPZ_EXT:
            d = np.load(path)
            images_as_np.append(d["slice"].astype(np.float32))
            if "mask" in d:
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

def extract_features_fn(examples: Dict, processor, backbone, dilation_radius: int = 8) -> Dict:
    """
    Return backbone features for a batch of samples as "pixel_values".

    Fast path — pre-cached patch_tokens in NPZ:
        If every path is an NPZ file that already contains a "patch_tokens" key
        (written by eval_pretrained.py), the tokens are loaded and masked-avg-pooled
        directly without running the backbone.  backbone may be None in this case.
        Also supports legacy "features" key (already-pooled vector).

    Slow path — run backbone:
        For PNG files or NPZ without cached tokens, the frozen backbone is run,
        then masked average pooling (or CLS fallback) is applied.

    The key name "pixel_values" is intentional: Classifier.forward expects it.
    """
    crop_size = _get_crop_size(processor)

    # ── Fast path: all tokens/features already cached in NPZ ──────────────────
    cached: list = []
    all_cached = True
    for path in examples["path"]:
        p = Path(path)
        if p.suffix.lower() == _NPZ_EXT:
            d = np.load(path)
            if "patch_tokens" in d:
                tokens = torch.from_numpy(d["patch_tokens"].copy())   # (N, D)
                if "mask" in d:
                    # Derive the crop_size that matches these tokens, not the processor's
                    N_tok = tokens.shape[0]
                    grid = int(N_tok ** 0.5)
                    token_crop_size = grid * _DINO_PATCH
                    mask_t = resize_mask(d["mask"], token_crop_size, dilation_radius)  # (1, 1, S, S)
                    pooled = _masked_avg_pool_tokens(
                        tokens.unsqueeze(0), mask_t.squeeze(0).unsqueeze(0)
                    )                                                  # (1, D)
                else:
                    # fallback: average all tokens equally
                    pooled = tokens.mean(dim=0, keepdim=True)          # (1, D)
                cached.append(pooled.squeeze(0))                       # (D,)
                continue
            elif "features" in d:
                # legacy: already-pooled vector
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
    masks_np: list = []

    for path in examples["path"]:
        p = Path(path)
        if p.suffix.lower() == _NPZ_EXT:
            d = np.load(path)
            images_as_np.append(d["slice"].astype(np.float32))
            masks_np.append(d["mask"] if "mask" in d else None)
        else:
            img = Image.open(path)
            if img.mode not in ("L", "F"):
                img = img.convert("L")
            images_as_np.append(np.array(img, dtype=np.float32))
            masks_np.append(None)

    processed = processor(images_as_np, return_tensors="pt")
    # Derive actual spatial size from pixel_values — processor.crop_size can differ
    actual_size = processed["pixel_values"].shape[-1]
    mask_tensors = [
        resize_mask(m, actual_size, dilation_radius) if m is not None else None
        for m in masks_np
    ]
    device = next(backbone.parameters()).device
    pixel_values = processed["pixel_values"].to(device)

    with torch.no_grad():
        outputs = backbone(pixel_values=pixel_values, output_hidden_states=False)

    if all(m is not None for m in mask_tensors):
        mask_batch = torch.cat(mask_tensors, dim=0).to(device)  # (B, 1, S, S)
        features = _masked_avg_pool(outputs.last_hidden_state, mask_batch).cpu()
    else:
        features = outputs.last_hidden_state[:, 0].cpu()    # CLS token fallback

    return {
        "pixel_values": features,
        "labels": torch.tensor(examples["target"], dtype=torch.long),
    }


# ── On-the-fly PyTorch Dataset for pre-cached patch_tokens ────────────────────

class PatchTokenDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset that reads patch_tokens + mask from NPZ files and applies
    masked-avg-pooling on-the-fly in __getitem__.

    Avoids the HF Dataset.map() pre-processing phase entirely: training starts
    immediately, pooling happens per-sample during dataloading (very fast with
    the vectorised resize_mask).
    """

    def __init__(self, paths: List[str], labels: List[int], dilation_radius: int = 8):
        self.paths           = paths
        self.labels          = labels
        self.dilation_radius = dilation_radius

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.paths[idx]
        label = self.labels[idx]
        d = np.load(path)

        tokens = torch.from_numpy(d["patch_tokens"].copy())  # (N, D)
        N_tok  = tokens.shape[0]
        grid   = int(N_tok ** 0.5)
        token_crop_size = grid * _DINO_PATCH

        if "mask" in d:
            mask_t = resize_mask(d["mask"], token_crop_size, self.dilation_radius)  # (1,1,S,S)
            pooled = _masked_avg_pool_tokens(
                tokens.unsqueeze(0), mask_t.squeeze(0).unsqueeze(0)
            ).squeeze(0)                                      # (D,)
        else:
            pooled = tokens.mean(dim=0)                       # (D,)

        return {
            "pixel_values": pooled,
            "labels": torch.tensor(label, dtype=torch.long),
        }


def _pool_npz_list(paths: List[str], labels: List[int], dilation_radius: int) -> torch.utils.data.TensorDataset:
    """
    Load all NPZ files, apply masked-avg-pooling, and return a TensorDataset
    with all features pre-loaded in RAM.  Fast startup, fast per-epoch.
    """
    features_list = []
    for path in tqdm(paths, desc="Pooling features", unit="ex", leave=False):
        d = np.load(path)
        tokens = torch.from_numpy(d["patch_tokens"].copy())  # (N, D)
        N_tok = tokens.shape[0]
        grid  = int(N_tok ** 0.5)
        token_crop_size = grid * _DINO_PATCH
        if "mask" in d:
            mask_t = resize_mask(d["mask"], token_crop_size, dilation_radius)  # (1,1,S,S)
            pooled = _masked_avg_pool_tokens(
                tokens.unsqueeze(0), mask_t.squeeze(0).unsqueeze(0)
            ).squeeze(0)                                      # (D,)
        else:
            pooled = tokens.mean(dim=0)
        features_list.append(pooled)
    features = torch.stack(features_list)                     # (N_samples, D)
    labels_t = torch.tensor(labels, dtype=torch.long)
    return torch.utils.data.TensorDataset(features, labels_t)


def build_patch_token_datasets(
    hf_train: Dataset,
    hf_val: Dataset,
    dilation_radius: int = 8,
    data_dir: str = "",
    cache_suffix: str = "",
) -> Tuple["_DictDataset", "_DictDataset"]:
    """
    Pool all patch_tokens into RAM tensors.  No disk I/O during training.

    Fast path: if {data_dir}/pooled_features[_{suffix}]_dil{N}.pt exists (written by
    cache_pooled_features.py), load it instantly and split into train/val by path.
    The cache_suffix parameter disambiguates caches from different backbones,
    e.g. cache_suffix="custom" → pooled_features_custom_dil8.pt.

    Slow path: read every NPZ, pool on the fly, stack into tensors (~1-3 min on NFS).
    Run cache_pooled_features.py once to avoid this on subsequent runs.
    """
    # ── Fast path: pre-computed .pt cache ─────────────────────────────────────
    suffix_part = f"_{cache_suffix}" if cache_suffix else ""
    pt_name = f"pooled_features{suffix_part}_dil{dilation_radius}.pt"
    pt_path = Path(data_dir) / pt_name if data_dir else None
    if pt_path and pt_path.exists():
        print(f"[cache] Loading pooled features from {pt_path}", flush=True)
        cache = torch.load(pt_path, weights_only=True)
        path_to_idx = {p: i for i, p in enumerate(cache["paths"])}

        def _make_ds(paths, targets):
            idxs = [path_to_idx[p] for p in paths]
            feats  = cache["features"][idxs]
            labels = torch.tensor(targets, dtype=torch.long)
            return _DictDataset(torch.utils.data.TensorDataset(feats, labels))

        train_ds = _make_ds(hf_train["path"], hf_train["target"])
        val_ds   = _make_ds(hf_val["path"],   hf_val["target"])
        n_train  = len(train_ds)
        n_val    = len(val_ds)
        D        = cache["features"].shape[1]
        mb       = pt_path.stat().st_size / 1e6
        print(f"[cache] {n_train} train + {n_val} val  ({D}d, {mb:.1f} MB file)", flush=True)
        return train_ds, val_ds

    # ── Slow path: read NPZ files one by one ──────────────────────────────────
    print("[cache] Pre-loading features into RAM (NPZ slow path)...", flush=True)
    suffix_flag = f" --cache_suffix {cache_suffix}" if cache_suffix else ""
    print("[cache] Tip: run  python -m classification_hf.cache_pooled_features "
          f"--data_dir {data_dir} --dilation_radius {dilation_radius}{suffix_flag}  to speed this up.", flush=True)
    train_tensor = _pool_npz_list(hf_train["path"], hf_train["target"], dilation_radius)
    val_tensor   = _pool_npz_list(hf_val["path"],   hf_val["target"],   dilation_radius)
    n_train, D = train_tensor.tensors[0].shape
    n_val       = val_tensor.tensors[0].shape[0]
    print(f"[cache] {n_train} train + {n_val} val features loaded ({D}d, "
          f"{(n_train + n_val) * D * 4 / 1e6:.1f} MB)", flush=True)
    return _DictDataset(train_tensor), _DictDataset(val_tensor)


class _DictDataset(torch.utils.data.Dataset):
    """Wrap a TensorDataset to return dicts expected by HF Trainer."""
    def __init__(self, tensor_ds: torch.utils.data.TensorDataset):
        self._ds = tensor_ds
    def __len__(self):
        return len(self._ds)
    def __getitem__(self, idx):
        features, label = self._ds[idx]
        return {"pixel_values": features, "labels": label}
