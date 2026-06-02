#!/usr/bin/env python3
"""
MS lesion segmentation: frozen foundation model backbone + CNN decoder.

Pipeline:
  1. Extract spatial features once → cached to disk as float16 memmap
  2. Train CNN decoder on cached features (no backbone needed)

Dataset : nih-ms-mp2rage (axial, matched slices only)
Protocol: 80/20 subject split, early stopping on val Dice (positive slices)

Usage:
    CUDA_VISIBLE_DEVICES=1 python train_lesion_seg.py [--model NAME]
    --model: Curia | DINOv3 | DINOv2+reg | MRI-CORE | all (default)
    --recompute_cache: force re-extraction even if cache exists
"""

from __future__ import annotations

import re, sys, random, argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from analyze_embeddings import load_model, _is_mricore, _MRICoreProcessor

# ── Config ────────────────────────────────────────────────────────────────────
DATASET    = "nih-ms-mp2rage"
DATA_ROOT  = Path("/home/ge.polymtl.ca/p123239/data_work/01_extracted_v2")
IMAGE_ROOT = DATA_ROOT / "image"
LABEL_ROOT = DATA_ROOT / "label_lesion"
OUT_DIR    = Path("seg_results")
CACHE_DIR  = Path("seg_feat_cache")

MODELS = {
    "Curia":      ("models/curia",                 512, 32, 768,  0),
    "DINOv3":     ("models/dinov3-vitl16",          224, 16, 1024, 4),
    "DINOv2+reg": ("models/dinov2-registers-large", 224, 16, 1024, 4),
    "MRI-CORE":   ("models/mricore",                256, 16, 256,  0),
}
# (model_path, input_size, feat_grid, feat_dim, n_reg)

SEED         = 42
VAL_FRAC     = 0.20
BATCH_SIZE   = 256    # decoder-only batches are cheap
EXTRACT_BS   = 64     # backbone extraction batch size
EPOCHS       = 100
LR           = 1e-3
PATIENCE     = 15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Key matching ──────────────────────────────────────────────────────────────
def path_to_key(p: str) -> str | None:
    parts = Path(p).stem.split("__")
    if len(parts) < 6:
        return None
    subj, _, orient, sidx, tidx, sp = parts[:6]
    return f"{subj}__{orient}__{sidx}__{tidx}__{sp}" if re.match(r"s\d+", sidx) else None


def build_label_path_index(dataset: str) -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for split in ("train", "val"):
        d = LABEL_ROOT / split / dataset
        if d.exists():
            for p in d.glob("*.png"):
                k = path_to_key(str(p))
                if k and k not in idx:
                    idx[k] = p
    return idx


# ── Feature extraction & caching ─────────────────────────────────────────────
class _ImageOnlyDataset(Dataset):
    def __init__(self, img_paths, processor, input_size, is_mricore):
        self.paths      = img_paths
        self.processor  = processor
        self.input_size = input_size
        self.is_mricore = is_mricore

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.paths[idx]))
        inputs = self.processor([img], return_tensors="pt")
        pv = inputs.pixel_values if hasattr(inputs, "pixel_values") else inputs["pixel_values"]
        return pv.squeeze(0)


class SpatialExtractor(nn.Module):
    def __init__(self, backbone, is_mricore, n_reg):
        super().__init__()
        self.backbone = backbone; self.is_mricore = is_mricore; self.n_reg = n_reg

    @torch.no_grad()
    def forward(self, x):
        if self.is_mricore:
            return self.backbone.image_encoder(x).float()
        hs = self.backbone(pixel_values=x).last_hidden_state.float()
        patches = hs[:, 1 + self.n_reg:, :]
        B, N, C = patches.shape
        h = int(N ** 0.5)
        return patches.permute(0, 2, 1).reshape(B, C, h, h)


def extract_and_cache(model_name: str, img_paths: list[Path],
                      recompute: bool = False) -> np.memmap:
    """Extract spatial features and save to disk. Returns memmap array (N, C, H, W) float16."""
    CACHE_DIR.mkdir(exist_ok=True)
    feat_path = CACHE_DIR / f"{model_name.replace('+','plus').replace(' ','_')}_features.npy"
    meta_path = CACHE_DIR / f"{model_name.replace('+','plus').replace(' ','_')}_paths.txt"

    model_path, input_size, feat_size, feat_dim, n_reg = MODELS[model_name]
    N = len(img_paths)
    shape = (N, feat_dim, feat_size, feat_size)

    if feat_path.exists() and meta_path.exists() and not recompute:
        # Verify same order
        cached_paths = meta_path.read_text().splitlines()
        if cached_paths == [str(p) for p in img_paths]:
            print(f"  Loading cached features from {feat_path}  ({feat_path.stat().st_size/1e9:.1f} GB)")
            return np.memmap(feat_path, dtype="float16", mode="r", shape=shape)
        print("  Cache exists but paths differ — re-extracting")

    gb = np.prod(shape) * 2 / 1e9
    print(f"  Extracting features → {feat_path}  ({gb:.1f} GB float16)…")

    backbone, processor = load_model(model_path, DEVICE)
    backbone.eval()
    is_mricore = isinstance(processor, _MRICoreProcessor)
    extractor  = SpatialExtractor(backbone, is_mricore, n_reg).to(DEVICE)

    mm = np.memmap(feat_path, dtype="float16", mode="w+", shape=shape)
    ds = _ImageOnlyDataset(img_paths, processor, input_size, is_mricore)
    loader = DataLoader(ds, batch_size=EXTRACT_BS, num_workers=4, pin_memory=False)

    i = 0
    with torch.amp.autocast("cuda"):
        for batch in tqdm(loader, desc=f"  {model_name} extraction"):
            feat = extractor(batch.to(DEVICE)).cpu().numpy().astype("float16")
            mm[i:i + len(feat)] = feat
            i += len(feat)
    mm.flush()

    meta_path.write_text("\n".join(str(p) for p in img_paths))
    del backbone, extractor
    torch.cuda.empty_cache()
    return np.memmap(feat_path, dtype="float16", mode="r", shape=shape)


# ── Cached dataset ────────────────────────────────────────────────────────────
class CachedSegDataset(Dataset):
    def __init__(self, indices, features_mm, lbl_paths, input_size):
        self.idx        = indices
        self.features   = features_mm
        self.lbl_paths  = lbl_paths
        self.input_size = input_size

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        feat = torch.from_numpy(self.features[j].astype("float32"))  # (C, h, w)

        lbl = np.array(Image.open(self.lbl_paths[j]).convert("L")).astype("float32") / 255.0
        lbl_t = torch.from_numpy(lbl).unsqueeze(0).unsqueeze(0)
        lbl_t = F.interpolate(lbl_t, (self.input_size, self.input_size), mode="nearest")
        lbl_binary = (lbl_t.squeeze() > 0.5).float()

        return feat, lbl_binary


# ── CNN Decoder ───────────────────────────────────────────────────────────────
class UpBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
    def forward(self, x): return self.conv(self.up(x))


class CNNDecoder(nn.Module):
    def __init__(self, in_c, feat_size, target_size):
        super().__init__()
        n_ups    = int(np.ceil(np.log2(target_size / feat_size)))
        channels = [in_c] + [max(32, in_c // (2 ** (i+1))) for i in range(n_ups)]
        self.ups  = nn.Sequential(*[UpBlock(channels[i], channels[i+1]) for i in range(n_ups)])
        self.head = nn.Conv2d(channels[-1], 1, 1)
        self.target_size = target_size

    def forward(self, x):
        x = self.ups(x)
        if x.shape[-1] != self.target_size:
            x = F.interpolate(x, (self.target_size, self.target_size),
                              mode="bilinear", align_corners=False)
        return self.head(x)


# ── Loss & metric ─────────────────────────────────────────────────────────────
def dice_bce_loss(logits, targets):
    probs  = torch.sigmoid(logits.squeeze(1))
    smooth = 1.0
    inter  = (probs * targets).sum((-2, -1))
    dice   = 1 - (2*inter + smooth) / (probs.sum((-2,-1)) + targets.sum((-2,-1)) + smooth)
    n_pos  = targets.sum(); n_neg = targets.numel() - n_pos
    pw     = torch.tensor([n_neg / max(n_pos.item(), 1)], device=logits.device)
    bce    = F.binary_cross_entropy_with_logits(logits.squeeze(1), targets, pos_weight=pw)
    return dice.mean() + bce


def compute_dice(logits, targets):
    preds  = (torch.sigmoid(logits.squeeze(1)) > 0.5).float()
    smooth = 1e-6
    inter  = (preds * targets).sum((-2, -1))
    dices  = (2*inter + smooth) / (preds.sum((-2,-1)) + targets.sum((-2,-1)) + smooth)
    has    = targets.sum((-2,-1)) > 0
    return dices[has].mean().item() if has.any() else float("nan"), dices.mean().item()


# ── Train / eval ──────────────────────────────────────────────────────────────
def train_epoch(decoder, loader, optimizer, scaler, epoch, epochs):
    decoder.train(); total = 0.0
    pbar = tqdm(loader, desc=f"  Ep {epoch:3d}/{epochs} train", leave=False)
    for feat, lbl in pbar:
        feat, lbl = feat.to(DEVICE), lbl.to(DEVICE)
        with torch.amp.autocast("cuda"):
            loss = dice_bce_loss(decoder(feat), lbl)
        optimizer.zero_grad(); scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        total += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total / len(loader)


@torch.no_grad()
def evaluate(decoder, loader):
    decoder.eval(); dp_list, da_list = [], []
    for feat, lbl in loader:
        feat, lbl = feat.to(DEVICE), lbl.to(DEVICE)
        with torch.amp.autocast("cuda"):
            dp, da = compute_dice(decoder(feat), lbl)
        if not np.isnan(dp): dp_list.append(dp)
        da_list.append(da)
    return (float(np.mean(dp_list)) if dp_list else 0.0), float(np.mean(da_list))


# ── Main ──────────────────────────────────────────────────────────────────────
def run_model(model_name: str, recompute_cache: bool = False):
    _, input_size, feat_size, feat_dim, _ = MODELS[model_name]
    print(f"\n{'='*60}\n[{model_name}]")

    label_index = build_label_path_index(DATASET)

    img_paths, lbl_paths, subjects = [], [], []
    for split in ("train", "val"):
        for img_path in sorted((IMAGE_ROOT / split / DATASET).glob("*.png")):
            if "desc-denoised_UNIT1" not in img_path.name:
                continue
            k = path_to_key(str(img_path))
            if k and k in label_index:
                img_paths.append(img_path)
                lbl_paths.append(label_index[k])
                subjects.append(k.split("__")[0])
    print(f"  {len(img_paths)} matched pairs")

    # ── Step 1: extract & cache features ──────────────────────────────────────
    features_mm = extract_and_cache(model_name, img_paths, recompute=recompute_cache)

    # ── Step 2: subject split ─────────────────────────────────────────────────
    unique_subj = list(set(subjects))
    rng = random.Random(SEED); rng.shuffle(unique_subj)
    n_val    = max(1, int(len(unique_subj) * VAL_FRAC))
    val_subj = set(unique_subj[:n_val])

    train_idx = [i for i, s in enumerate(subjects) if s not in val_subj]
    val_idx   = [i for i, s in enumerate(subjects) if s in val_subj]
    print(f"  Train: {len(train_idx)} slices  Val: {len(val_idx)} slices")

    train_ds = CachedSegDataset(train_idx, features_mm, lbl_paths, input_size)
    val_ds   = CachedSegDataset(val_idx,   features_mm, lbl_paths, input_size)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=False)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=False)

    # ── Step 3: train decoder ─────────────────────────────────────────────────
    decoder   = CNNDecoder(feat_dim, feat_size, input_size).to(DEVICE)
    n_params  = sum(p.numel() for p in decoder.parameters())
    print(f"  Decoder params: {n_params:,}")

    optimizer = torch.optim.Adam(decoder.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", 0.5, patience=5, min_lr=1e-6)
    scaler    = torch.amp.GradScaler("cuda")

    best_dice, best_state, no_improve = 0.0, None, 0

    for epoch in range(1, EPOCHS + 1):
        loss   = train_epoch(decoder, train_loader, optimizer, scaler, epoch, EPOCHS)
        dp, da = evaluate(decoder, val_loader)
        scheduler.step(dp)
        print(f"  Ep {epoch:3d}/{EPOCHS}  loss={loss:.4f}  val_dice_pos={dp:.4f}  val_dice_all={da:.4f}")

        if dp > best_dice + 1e-4:
            best_dice  = dp
            best_state = {k: v.cpu().clone() for k, v in decoder.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    if best_state:
        decoder.load_state_dict(best_state)

    dp_final, da_final = evaluate(decoder, val_loader)
    print(f"\n  Best Dice (positive slices): {best_dice:.4f}")

    save_path = OUT_DIR / f"decoder_{model_name.replace(' ','_').replace('+','plus')}.pt"
    torch.save(decoder.state_dict(), save_path)
    del decoder; torch.cuda.empty_cache()

    return {"model": model_name, "dice_pos": round(best_dice, 4), "dice_all": round(da_final, 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all")
    parser.add_argument("--recompute_cache", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    to_run = list(MODELS.keys()) if args.model == "all" else [args.model]
    results = [run_model(n, args.recompute_cache) for n in to_run]

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "seg_results.csv", index=False)
    print(f"\n{'='*60}\n{df.to_string(index=False)}")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2166ac", "#d6604d", "#4dac26", "#8073ac"]
    x = np.arange(len(df))
    ax.bar(x, df["dice_pos"], color=colors[:len(df)], alpha=0.85, width=0.5)
    for xi, row in zip(x, df.itertuples()):
        ax.text(xi, row.dice_pos + 0.005, f"{row.dice_pos:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(df["model"], fontsize=10)
    ax.set_ylabel("Dice score (positive slices)", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"MS lesion segmentation — {DATASET}\nFrozen backbone + CNN decoder",
                 fontsize=11, fontweight="bold")
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    fig.patch.set_facecolor("white"); plt.tight_layout()
    plt.savefig(OUT_DIR / "seg_results.png", dpi=150, bbox_inches="tight")
    print(f"Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
