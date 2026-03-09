import random
import sys
import importlib
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .config import TrainConfig
from .dataset import (
    NpzSegmentationDataset,
    build_datasets,
    build_npz_datasets,
    build_npz_train_dataloader,
    build_npz_val_dataloader,
    build_train_dataloader,
    make_sliding_positions,
    normalize_image_array,
    overlap_pct_to_pixels,
    pad_to_min_hw,
)
from .losses import compute_dice_score, dice_loss_with_logits
from .model import FrozenBackboneWithSegHead


def safe_import_wandb():
    wandb = importlib.import_module("wandb")
    if callable(getattr(wandb, "init", None)):
        return wandb

    cwd = str(Path.cwd().resolve())
    removed: list[tuple[int, str]] = []
    for i in reversed(range(len(sys.path))):
        p = sys.path[i]
        p_resolved = cwd if p == "" else str(Path(p).resolve())
        if p == "" or p_resolved == cwd:
            removed.append((i, p))
            sys.path.pop(i)

    try:
        importlib.invalidate_caches()
        sys.modules.pop("wandb", None)
        wandb = importlib.import_module("wandb")
    finally:
        for i, p in sorted(removed, key=lambda t: t[0]):
            sys.path.insert(i, p)

    if callable(getattr(wandb, "init", None)):
        return wandb

    wandb_file = getattr(wandb, "__file__", None)
    raise RuntimeError(
        "Imported 'wandb' module has no 'init'. "
        f"Resolved module file: {wandb_file}. "
        "Possible local module shadowing (e.g., a local 'wandb' folder/file) or broken installation."
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_val_dice: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_dice": best_val_dice,
    }
    torch.save(ckpt, path)


def autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=enabled)
    return nullcontext()


def normalize_overlay_sizes(examples: list[dict]) -> list[dict]:
    if not examples:
        return examples

    max_h = max(ex["overlay"].shape[0] for ex in examples)
    max_w = max(ex["overlay"].shape[1] for ex in examples)
    out = []

    for ex in examples:
        overlay = ex["overlay"]
        h, w = overlay.shape[:2]
        if h == max_h and w == max_w:
            out.append(ex)
            continue

        pad_h = max_h - h
        pad_w = max_w - w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        padded = np.pad(overlay, ((top, bottom), (left, right), (0, 0)), mode="constant", constant_values=0)

        ex_new = dict(ex)
        ex_new["overlay"] = padded
        out.append(ex_new)

    return out


def make_overlay_panel(image_2d: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    image_u8 = np.clip(image_2d, 0, 255).astype(np.uint8)
    base_rgb = np.stack([image_u8, image_u8, image_u8], axis=-1).astype(np.float32)

    gt = gt_mask.astype(bool)
    pred = pred_mask.astype(bool)

    overlay = base_rgb.copy()

    alpha = 0.45
    gt_only = gt & (~pred)
    pred_only = pred & (~gt)
    both = gt & pred

    overlay[gt_only, 1] = (1 - alpha) * overlay[gt_only, 1] + alpha * 255.0
    overlay[gt_only, 0] = (1 - alpha) * overlay[gt_only, 0]
    overlay[gt_only, 2] = (1 - alpha) * overlay[gt_only, 2]

    overlay[pred_only, 0] = (1 - alpha) * overlay[pred_only, 0] + alpha * 255.0
    overlay[pred_only, 1] = (1 - alpha) * overlay[pred_only, 1]
    overlay[pred_only, 2] = (1 - alpha) * overlay[pred_only, 2]

    overlay[both, 0] = (1 - alpha) * overlay[both, 0] + alpha * 255.0
    overlay[both, 1] = (1 - alpha) * overlay[both, 1] + alpha * 255.0
    overlay[both, 2] = (1 - alpha) * overlay[both, 2]

    return np.clip(overlay, 0, 255).astype(np.uint8)


def run_train_epoch_from_tokens(
    model: FrozenBackboneWithSegHead,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    bce_weight: float,
    dice_weight: float,
    amp: bool,
    target_hw: tuple[int, int],
) -> tuple[float, float]:
    """Training epoch using pre-cached patch tokens — backbone is never called."""
    model.train()
    bce_loss_fn  = nn.BCEWithLogitsLoss()
    running_loss = 0.0
    running_dice = 0.0
    num_batches  = 0

    pbar = tqdm(loader, desc="train (tokens)", leave=False)
    for patch_tokens, masks in pbar:
        patch_tokens = patch_tokens.to(device, non_blocking=True)
        masks        = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device=device, enabled=amp):
            logits = model.forward_from_tokens(patch_tokens, target_hw)
            bce    = bce_loss_fn(logits, masks)
            dloss  = dice_loss_with_logits(logits, masks)
            loss   = bce_weight * bce + dice_weight * dloss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_dice    = compute_dice_score(logits.detach(), masks)
        running_loss += loss.detach().item()
        running_dice += batch_dice
        num_batches  += 1

        pbar.set_postfix(loss=f"{running_loss / num_batches:.4f}",
                         dice=f"{running_dice / num_batches:.4f}")

    if num_batches == 0:
        return 0.0, 0.0
    return running_loss / num_batches, running_dice / num_batches


@torch.no_grad()
def run_full_image_eval_npz(
    model: FrozenBackboneWithSegHead,
    ds: NpzSegmentationDataset,
    device: torch.device,
    bce_weight: float,
    dice_weight: float,
    amp: bool,
    target_hw: tuple[int, int],
    desc: str = "eval",
) -> tuple[float, float]:
    """Full-image eval for NpzSegmentationDataset.

    Fast path (tokens cached): loads patch_tokens from NPZ, calls
    forward_from_tokens() — backbone is never used.

    Slow path (no tokens): falls back to predict_full_image_detiled()
    which runs the full backbone on the raw slice.
    """
    model.eval()
    bce_loss_fn  = nn.BCEWithLogitsLoss()
    running_loss = 0.0
    running_dice = 0.0
    num_samples  = 0

    pbar = tqdm(range(len(ds)), desc=desc, leave=False)
    for idx in pbar:
        if ds.has_tokens:
            d            = np.load(ds.npz_paths[idx])
            patch_tokens = torch.from_numpy(d[ds.token_key].astype(np.float32))
            mask_np      = (d["mask"].astype(np.float32) > 0)

            pt = patch_tokens.unsqueeze(0).to(device)
            with autocast_context(device=device, enabled=amp):
                logits = model.forward_from_tokens(pt, target_hw)  # (1,1,H,W)
            logits = logits.cpu()

            mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float()
            mask_t = F.interpolate(mask_t, size=target_hw, mode="nearest")
        else:
            image, mask_np = ds.load_raw_pair(idx)
            logits = predict_full_image_detiled(
                model=model,
                image_2d=image,
                tile_size=ds.image_size,
                tile_overlap_pct=ds.tile_overlap_pct,
                tile_threshold=ds.tile_threshold,
                device=device,
                amp=amp,
                tile_batch_size=4,
            ).unsqueeze(0)
            mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float()

        bce   = bce_loss_fn(logits, mask_t)
        dloss = dice_loss_with_logits(logits, mask_t)
        loss  = bce_weight * bce + dice_weight * dloss
        dice  = compute_dice_score(logits, mask_t)

        running_loss += loss.item()
        running_dice += dice
        num_samples  += 1
        pbar.set_postfix(loss=f"{running_loss / num_samples:.4f}",
                         dice=f"{running_dice / num_samples:.4f}")

    if num_samples == 0:
        return 0.0, 0.0
    return running_loss / num_samples, running_dice / num_samples


def run_train_epoch(
    model: FrozenBackboneWithSegHead,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    bce_weight: float,
    dice_weight: float,
    amp: bool,
) -> Tuple[float, float]:
    model.train()
    model.backbone.eval()

    bce_loss_fn = nn.BCEWithLogitsLoss()
    running_loss = 0.0
    running_dice = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device=device, enabled=amp):
            logits = model(images)
            bce = bce_loss_fn(logits, masks)
            dloss = dice_loss_with_logits(logits, masks)
            loss = bce_weight * bce + dice_weight * dloss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_dice = compute_dice_score(logits.detach(), masks)
        running_loss += loss.detach().item()
        running_dice += batch_dice
        num_batches += 1

        pbar.set_postfix(loss=f"{running_loss / num_batches:.4f}", dice=f"{running_dice / num_batches:.4f}")

    if num_batches == 0:
        return 0.0, 0.0
    return running_loss / num_batches, running_dice / num_batches


@torch.no_grad()
def predict_full_image_detiled(
    model: FrozenBackboneWithSegHead,
    image_2d: np.ndarray,
    tile_size: int,
    tile_overlap_pct: float,
    tile_threshold: int,
    device: torch.device,
    amp: bool,
    tile_batch_size: int,
) -> torch.Tensor:
    h, w = image_2d.shape

    # Normalize the full image once, before any tiling.
    # Tiles are extracted from the normalized array, so every tile shares the same
    # image-level statistics — consistent with __getitem__ in the dataset.
    # Padding fills with 0, which equals the image mean after z-score normalization.
    image_norm = normalize_image_array(image_2d / 255.0)

    must_tile = max(h, w) > tile_threshold

    if not must_tile:
        # Pad to tile_size so the backbone sees the same input size as during training.
        image_pad = pad_to_min_hw(image_norm, tile_size, tile_size, fill=0.0)
        image_t = torch.from_numpy(image_pad).unsqueeze(0).unsqueeze(0).float().to(device)
        with autocast_context(device=device, enabled=amp):
            logits = model(image_t)
        # Crop back to original spatial dimensions
        return logits[:, :, :h, :w].squeeze(0).cpu()

    overlap_px = overlap_pct_to_pixels(tile_size=tile_size, overlap_pct=tile_overlap_pct)

    image_pad = pad_to_min_hw(image_norm, tile_size, tile_size, fill=0.0)
    hp, wp = image_pad.shape

    xs = make_sliding_positions(wp, tile_size, overlap_px)
    ys = make_sliding_positions(hp, tile_size, overlap_px)

    logits_sum = torch.zeros((1, hp, wp), dtype=torch.float32, device=device)
    logits_count = torch.zeros((1, hp, wp), dtype=torch.float32, device=device)

    coords = [(x0, y0) for y0 in ys for x0 in xs]

    for start in range(0, len(coords), tile_batch_size):
        batch_coords = coords[start : start + tile_batch_size]
        tile_tensors = []
        for x0, y0 in batch_coords:
            tile = image_pad[y0 : y0 + tile_size, x0 : x0 + tile_size]
            tile_t = torch.from_numpy(tile.copy()).unsqueeze(0)  # already normalized
            tile_tensors.append(tile_t)

        tiles_batch = torch.stack(tile_tensors, dim=0).to(device)

        with autocast_context(device=device, enabled=amp):
            logits_batch = model(tiles_batch)

        for i, (x0, y0) in enumerate(batch_coords):
            logit_tile = logits_batch[i, 0]
            logits_sum[0, y0 : y0 + tile_size, x0 : x0 + tile_size] += logit_tile
            logits_count[0, y0 : y0 + tile_size, x0 : x0 + tile_size] += 1.0

    logits_full = logits_sum / torch.clamp_min(logits_count, 1.0)
    logits_full = logits_full[:, :h, :w]
    return logits_full.cpu()


@torch.no_grad()
def run_full_image_eval(
    model: FrozenBackboneWithSegHead,
    ds,
    device: torch.device,
    bce_weight: float,
    dice_weight: float,
    amp: bool,
    tile_batch_size: int,
    desc: str = "eval",
) -> Tuple[float, float]:
    """Full-image evaluation: tile-stitch each image then compute loss/dice.

    Used for both train and val metrics so they are directly comparable —
    both operate on reconstructed full images, not on individual tiles.
    """
    model.eval()
    bce_loss_fn = nn.BCEWithLogitsLoss()
    running_loss = 0.0
    running_dice = 0.0
    num_samples = 0

    pbar = tqdm(range(len(ds.pairs)), desc=desc, leave=False, total=len(ds.pairs))
    for pair_idx in pbar:
        image, mask = ds.load_raw_pair(pair_idx)

        logits = predict_full_image_detiled(
            model=model,
            image_2d=image,
            tile_size=ds.image_size,
            tile_overlap_pct=ds.tile_overlap_pct,
            tile_threshold=ds.tile_threshold,
            device=device,
            amp=amp,
            tile_batch_size=tile_batch_size,
        )

        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        bce = bce_loss_fn(logits.unsqueeze(0), mask_t)
        dloss = dice_loss_with_logits(logits.unsqueeze(0), mask_t)
        loss = bce_weight * bce + dice_weight * dloss
        dice = compute_dice_score(logits.unsqueeze(0), mask_t)

        running_loss += loss.item()
        running_dice += dice
        num_samples += 1
        pbar.set_postfix(loss=f"{running_loss / num_samples:.4f}", dice=f"{running_dice / num_samples:.4f}")

    if num_samples == 0:
        return 0.0, 0.0
    return running_loss / num_samples, running_dice / num_samples


@torch.no_grad()
def capture_val_overlays(
    model: FrozenBackboneWithSegHead,
    val_ds,
    device: torch.device,
    amp: bool,
    tile_batch_size: int,
    max_examples: int,
) -> list[dict]:
    """Full-image reconstruction used only for W&B overlay visualisation."""
    model.eval()
    examples: list[dict] = []
    fallback_examples: list[dict] = []

    for pair_idx in range(len(val_ds.pairs)):
        if len(examples) >= max_examples and len(fallback_examples) >= max_examples:
            break

        img_path = val_ds.pairs[pair_idx][0]
        image, mask = val_ds.load_raw_pair(pair_idx)

        logits = predict_full_image_detiled(
            model=model,
            image_2d=image,
            tile_size=val_ds.image_size,
            tile_overlap_pct=val_ds.tile_overlap_pct,
            tile_threshold=val_ds.tile_threshold,
            device=device,
            amp=amp,
            tile_batch_size=tile_batch_size,
        )

        pred_mask = (torch.sigmoid(logits).squeeze(0).numpy() > 0.5).astype(np.uint8)
        gt_mask = mask.astype(np.uint8)
        panel = make_overlay_panel(image_2d=image, gt_mask=gt_mask, pred_mask=pred_mask)
        gt_pixels = int(gt_mask.sum())
        pred_pixels = int(pred_mask.sum())
        dice = compute_dice_score(
            logits.unsqueeze(0),
            torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float(),
        )

        sample = {
            "name": img_path.name,
            "overlay": panel,
            "dice": float(dice),
            "gt_pixels": gt_pixels,
            "pred_pixels": pred_pixels,
        }

        if gt_pixels > 0 or pred_pixels > 0:
            if len(examples) < max_examples:
                examples.append(sample)
        elif len(fallback_examples) < max_examples:
            fallback_examples.append(sample)

    if len(examples) < max_examples:
        need = max_examples - len(examples)
        examples.extend(fallback_examples[:need])

    return examples


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Dataset / dataloader ──────────────────────────────────────────────────
    use_npz   = bool(cfg.npz_train_dir and cfg.npz_val_dir)
    target_hw = (cfg.image_size, cfg.image_size)

    if use_npz:
        train_ds, val_ds = build_npz_datasets(cfg)
        train_loader     = build_npz_train_dataloader(cfg, train_ds)
        fast_path        = train_ds.has_tokens
        print(f"NPZ dataset: {'fast path (cached tokens)' if fast_path else 'slow path (backbone)'}")
    else:
        train_ds, val_ds = build_datasets(cfg)
        train_loader     = build_train_dataloader(cfg, train_ds)
        fast_path        = False

    model = FrozenBackboneWithSegHead(cfg.model_dir).to(device)
    optimizer = torch.optim.AdamW(model.seg_head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    wandb_run = None
    if cfg.use_wandb and cfg.wandb_mode != "disabled":
        try:
            wandb = safe_import_wandb()

            wandb_run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=cfg.wandb_run_name,
                config=asdict(cfg),
                mode=cfg.wandb_mode,
            )
        except Exception as exc:
            print(f"[wandb] disabled due to init error: {exc}")
            wandb_run = None

    best_val_dice = -1.0
    history_path = output_dir / "history.csv"
    if not history_path.exists():
        history_path.write_text("epoch,train_loss,train_dice,val_loss,val_dice\n")

    print("Training config:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")
    print(f"  device: {device}")

    try:
        for epoch in range(1, cfg.epochs + 1):
            if fast_path:
                run_train_epoch_from_tokens(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    device=device,
                    scaler=scaler,
                    bce_weight=cfg.bce_weight,
                    dice_weight=cfg.dice_weight,
                    amp=cfg.amp,
                    target_hw=target_hw,
                )
            else:
                run_train_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    device=device,
                    scaler=scaler,
                    bce_weight=cfg.bce_weight,
                    dice_weight=cfg.dice_weight,
                    amp=cfg.amp,
                )

            if use_npz:
                train_loss, train_dice = run_full_image_eval_npz(
                    model=model,
                    ds=train_ds,
                    device=device,
                    bce_weight=cfg.bce_weight,
                    dice_weight=cfg.dice_weight,
                    amp=cfg.amp,
                    target_hw=target_hw,
                    desc="train-eval",
                )
                val_loss, val_dice = run_full_image_eval_npz(
                    model=model,
                    ds=val_ds,
                    device=device,
                    bce_weight=cfg.bce_weight,
                    dice_weight=cfg.dice_weight,
                    amp=cfg.amp,
                    target_hw=target_hw,
                    desc="val",
                )
            else:
                train_loss, train_dice = run_full_image_eval(
                    model=model,
                    ds=train_ds,
                    device=device,
                    bce_weight=cfg.bce_weight,
                    dice_weight=cfg.dice_weight,
                    amp=cfg.amp,
                    tile_batch_size=cfg.batch_size,
                    desc="train-eval",
                )

                val_loss, val_dice = run_full_image_eval(
                    model=model,
                    ds=val_ds,
                    device=device,
                    bce_weight=cfg.bce_weight,
                    dice_weight=cfg.dice_weight,
                    amp=cfg.amp,
                    tile_batch_size=cfg.batch_size,
                    desc="val",
                )

            want_overlays = (
                wandb_run is not None
                and cfg.wandb_log_val_images
                and cfg.wandb_val_images_count > 0
                and (epoch % max(1, cfg.wandb_val_images_every) == 0)
            )
            val_examples = (
                capture_val_overlays(
                    model=model,
                    val_ds=val_ds,
                    device=device,
                    amp=cfg.amp,
                    tile_batch_size=cfg.batch_size,
                    max_examples=cfg.wandb_val_images_count,
                )
                if want_overlays
                else []
            )

            print(
                f"Epoch {epoch:03d}/{cfg.epochs:03d} | "
                f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} | "
                f"val_loss={val_loss:.4f} val_dice={val_dice:.4f}"
            )

            with history_path.open("a") as f:
                f.write(f"{epoch},{train_loss:.6f},{train_dice:.6f},{val_loss:.6f},{val_dice:.6f}\n")

            if wandb_run is not None:
                payload = {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/dice": train_dice,
                    "val/loss": val_loss,
                    "val/dice": val_dice,
                    "best/val_dice": max(best_val_dice, val_dice),
                    "val/overlays_count": 0,
                }

                if want_overlays and len(val_examples) > 0:
                    wandb = safe_import_wandb()

                    val_examples = normalize_overlay_sizes(val_examples)
                    payload["val/overlays_count"] = len(val_examples)

                    payload["val/overlays"] = [
                        wandb.Image(
                            ex["overlay"],
                            caption=(
                                f"{ex['name']} | dice={ex['dice']:.4f} "
                                f"| gt_px={ex.get('gt_pixels', -1)} | pred_px={ex.get('pred_pixels', -1)}"
                            ),
                        )
                        for ex in val_examples
                    ]
                    print(f"[wandb] overlays logged: {len(val_examples)}")

                wandb_run.log(payload)

            if epoch % cfg.save_every == 0:
                save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, best_val_dice)

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, best_val_dice)
                print(f"  New best checkpoint at epoch {epoch} (val_dice={val_dice:.4f})")

    finally:
        if wandb_run is not None:
            wandb_run.finish()

    print(f"Done. Best val Dice: {best_val_dice:.4f}")
