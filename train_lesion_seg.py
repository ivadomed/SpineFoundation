#!/usr/bin/env python3
"""
MS lesion segmentation: frozen foundation model backbone + CNN decoder.

Dataset : nih-ms-mp2rage (axial orientation, matched slices only)
Protocol: frozen backbone, trained CNN decoder, 80/20 subject split,
          early stopping on val Dice (positive slices), Dice+BCE loss.

Usage:
    CUDA_VISIBLE_DEVICES=1 /path/to/python train_lesion_seg.py [--model NAME]

    --model: one of Curia, DINOv3, DINOv2+reg, MRI-CORE, or "all" (default)
"""

from __future__ import annotations

import re
import sys
import random
import argparse
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

MODELS = {
    "Curia":        ("models/curia",                  512, 32, 768,  0),
    "DINOv3":       ("models/dinov3-vitl16",           224, 16, 1024, 4),
    "DINOv2+reg":   ("models/dinov2-registers-large",  224, 16, 1024, 4),
    "MRI-CORE":     ("models/mricore",                 256, 16, 256,  0),
}
# (model_path, input_size, feat_grid, feat_dim, n_reg)

SEED       = 42
VAL_FRAC   = 0.20
BATCH_SIZE = 16
EPOCHS     = 100
LR         = 1e-3
PATIENCE   = 15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Key matching ──────────────────────────────────────────────────────────────
def path_to_key(path_str: str) -> str | None:
    parts = Path(path_str).stem.split("__")
    if len(parts) < 6:
        return None
    subject, _, orientation, sidx, tidx, spacing = parts[:6]
    if not re.match(r"s\d+", sidx):
        return None
    return f"{subject}__{orientation}__{sidx}__{tidx}__{spacing}"


def build_label_path_index(dataset: str) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for split in ("train", "val"):
        d = LABEL_ROOT / split / dataset
        if d.exists():
            for p in d.glob("*.png"):
                key = path_to_key(str(p))
                if key and key not in index:
                    index[key] = p
    return index


# ── Dataset ───────────────────────────────────────────────────────────────────
class LesionSegDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]],
                 processor, input_size: int, is_mricore: bool):
        self.pairs      = pairs
        self.processor  = processor
        self.input_size = input_size
        self.is_mricore = is_mricore

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, lbl_path = self.pairs[idx]

        img = np.array(Image.open(img_path))

        inputs = self.processor([img], return_tensors="pt")
        if hasattr(inputs, "pixel_values"):
            pv = inputs.pixel_values.squeeze(0)
        else:
            pv = inputs["pixel_values"].squeeze(0)

        lbl = np.array(Image.open(lbl_path).convert("L")).astype(np.float32) / 255.0
        lbl_t = torch.from_numpy(lbl).unsqueeze(0).unsqueeze(0)
        lbl_t = F.interpolate(lbl_t, size=(self.input_size, self.input_size),
                              mode="nearest").squeeze(0).squeeze(0)
        lbl_binary = (lbl_t > 0.5).float()

        return pv, lbl_binary


# ── Backbone spatial extractor (always frozen) ────────────────────────────────
class SpatialExtractor(nn.Module):
    def __init__(self, backbone, is_mricore: bool, n_reg: int):
        super().__init__()
        self.backbone   = backbone
        self.is_mricore = is_mricore
        self.n_reg      = n_reg

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_mricore:
            return self.backbone.image_encoder(x).float()

        out     = self.backbone(pixel_values=x)
        hs      = out.last_hidden_state.float()      # (B, 1+n_reg+N, C)
        patches = hs[:, 1 + self.n_reg:, :]          # (B, N, C)
        B, N, C = patches.shape
        h = w = int(N ** 0.5)
        return patches.permute(0, 2, 1).reshape(B, C, h, w)


# ── CNN Decoder ───────────────────────────────────────────────────────────────
class UpBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(self.up(x))


class CNNDecoder(nn.Module):
    def __init__(self, in_c: int, feat_size: int, target_size: int):
        super().__init__()
        n_ups = int(np.ceil(np.log2(target_size / feat_size)))
        channels = [in_c] + [max(32, in_c // (2 ** (i + 1))) for i in range(n_ups)]
        self.ups  = nn.Sequential(*[UpBlock(channels[i], channels[i+1]) for i in range(n_ups)])
        self.head = nn.Conv2d(channels[-1], 1, 1)
        self.target_size = target_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ups(x)
        if x.shape[-1] != self.target_size:
            x = F.interpolate(x, (self.target_size, self.target_size),
                              mode="bilinear", align_corners=False)
        return self.head(x)   # (B, 1, H, W) logits


# ── Loss & metric ─────────────────────────────────────────────────────────────
def dice_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits.squeeze(1))
    smooth = 1.0
    inter  = (probs * targets).sum((-2, -1))
    dice   = 1 - (2 * inter + smooth) / (probs.sum((-2, -1)) + targets.sum((-2, -1)) + smooth)

    n_pos = targets.sum()
    n_neg = targets.numel() - n_pos
    pw    = torch.tensor([n_neg / max(n_pos.item(), 1)], device=logits.device)
    bce   = F.binary_cross_entropy_with_logits(logits.squeeze(1), targets, pos_weight=pw)

    return dice.mean() + bce


def compute_dice(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    """Returns (dice_positive_slices, dice_all_slices)."""
    preds  = (torch.sigmoid(logits.squeeze(1)) > 0.5).float()
    smooth = 1e-6
    inter  = (preds * targets).sum((-2, -1))
    dices  = (2 * inter + smooth) / (preds.sum((-2, -1)) + targets.sum((-2, -1)) + smooth)
    has_lesion = targets.sum((-2, -1)) > 0
    dice_pos = dices[has_lesion].mean().item() if has_lesion.any() else float("nan")
    return dice_pos, dices.mean().item()


# ── Train / eval loops ────────────────────────────────────────────────────────
def train_epoch(extractor, decoder, loader, optimizer, scaler):
    decoder.train()
    total = 0.0
    for pv, lbl in loader:
        pv, lbl = pv.to(DEVICE), lbl.to(DEVICE)
        with torch.amp.autocast("cuda"):
            loss = dice_bce_loss(decoder(extractor(pv)), lbl)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(extractor, decoder, loader):
    decoder.eval()
    dp_list, da_list = [], []
    for pv, lbl in loader:
        pv, lbl = pv.to(DEVICE), lbl.to(DEVICE)
        with torch.amp.autocast("cuda"):
            logits = decoder(extractor(pv))
        dp, da = compute_dice(logits, lbl)
        if not np.isnan(dp):
            dp_list.append(dp)
        da_list.append(da)
    return float(np.mean(dp_list)) if dp_list else 0.0, float(np.mean(da_list))


# ── Main ──────────────────────────────────────────────────────────────────────
def run_model(model_name: str):
    model_path, input_size, feat_size, feat_dim, n_reg = MODELS[model_name]

    print(f"\n{'='*60}")
    print(f"[{model_name}]  input={input_size}  feat={feat_size}x{feat_size}  dim={feat_dim}  n_reg={n_reg}")

    # Build label index
    label_index = build_label_path_index(DATASET)

    # Collect image-label pairs
    pairs, subjects = [], []
    for split in ("train", "val"):
        for img_path in (IMAGE_ROOT / split / DATASET).glob("*.png"):
            key = path_to_key(str(img_path))
            if key and key in label_index:
                pairs.append((img_path, label_index[key]))
                subjects.append(key.split("__")[0])
    print(f"  Matched: {len(pairs)} image-label pairs")

    # Subject-based 80/20 split
    unique_subj = list(set(subjects))
    rng = random.Random(SEED)
    rng.shuffle(unique_subj)
    n_val      = max(1, int(len(unique_subj) * VAL_FRAC))
    val_subj   = set(unique_subj[:n_val])
    train_pairs = [(p, l) for (p, l), s in zip(pairs, subjects) if s not in val_subj]
    val_pairs   = [(p, l) for (p, l), s in zip(pairs, subjects) if s in val_subj]
    print(f"  Train: {len(train_pairs)} slices ({len(unique_subj) - n_val} subjects)")
    print(f"  Val:   {len(val_pairs)} slices ({n_val} subjects)")

    # Load model
    print("  Loading backbone...")
    backbone, processor = load_model(model_path, DEVICE)
    backbone.eval()
    is_mricore = isinstance(processor, _MRICoreProcessor)

    extractor = SpatialExtractor(backbone, is_mricore, n_reg).to(DEVICE)
    decoder   = CNNDecoder(feat_dim, feat_size, input_size).to(DEVICE)
    n_params  = sum(p.numel() for p in decoder.parameters())
    print(f"  Decoder params: {n_params:,}")

    train_ds = LesionSegDataset(train_pairs, processor, input_size, is_mricore)
    val_ds   = LesionSegDataset(val_pairs,   processor, input_size, is_mricore)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    best_dice_pos = 0.0
    best_state    = None
    no_improve    = 0

    for epoch in range(1, EPOCHS + 1):
        loss      = train_epoch(extractor, decoder, train_loader, optimizer, scaler)
        dp, da    = evaluate(extractor, decoder, val_loader)
        scheduler.step(dp)
        print(f"  Ep {epoch:3d}/{EPOCHS}  loss={loss:.4f}  "
              f"val_dice_pos={dp:.4f}  val_dice_all={da:.4f}")

        if dp > best_dice_pos + 1e-4:
            best_dice_pos = dp
            best_state = {k: v.cpu().clone() for k, v in decoder.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    if best_state:
        decoder.load_state_dict(best_state)

    dp_final, da_final = evaluate(extractor, decoder, val_loader)
    print(f"\n  Best Dice (positive slices): {best_dice_pos:.4f}")
    print(f"  Final Dice all slices:       {da_final:.4f}")

    save_path = OUT_DIR / f"decoder_{model_name.replace(' ', '_').replace('+', 'plus')}.pt"
    torch.save(decoder.state_dict(), save_path)
    print(f"  Decoder saved to {save_path}")

    del backbone, extractor, decoder
    torch.cuda.empty_cache()

    return {"model": model_name, "dice_pos": round(best_dice_pos, 4),
            "dice_all": round(da_final, 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all",
                        help="Model name or 'all'")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    to_run = list(MODELS.keys()) if args.model == "all" else [args.model]
    results = []

    for name in to_run:
        results.append(run_model(name))

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "seg_results.csv", index=False)
    print(f"\n{'='*60}")
    print("Results:")
    print(df.to_string(index=False))

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2166ac", "#d6604d", "#4dac26", "#8073ac"]
    x = np.arange(len(df))
    ax.bar(x, df["dice_pos"], color=colors[:len(df)], alpha=0.85, width=0.5)
    for xi, row in zip(x, df.itertuples()):
        ax.text(xi, row.dice_pos + 0.005, f"{row.dice_pos:.3f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(df["model"], fontsize=10)
    ax.set_ylabel("Dice score (positive slices)", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"MS lesion segmentation — {DATASET}\nFrozen backbone + CNN decoder",
                 fontsize=11, fontweight="bold")
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "seg_results.png", dpi=150, bbox_inches="tight")
    print(f"Plot saved to {OUT_DIR}/seg_results.png")


if __name__ == "__main__":
    main()
