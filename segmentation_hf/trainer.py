import random
from dataclasses import asdict
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainConfig
from .dataset import build_dataloaders
from .losses import compute_dice_score, dice_loss_with_logits
from .model import FrozenBackboneWithSegHead


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


def run_epoch(
    model: FrozenBackboneWithSegHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    train: bool,
    bce_weight: float,
    dice_weight: float,
    amp: bool,
) -> Tuple[float, float]:
    if train:
        model.train()
        model.backbone.eval()
    else:
        model.eval()

    bce_loss_fn = nn.BCEWithLogitsLoss()
    running_loss = 0.0
    running_dice = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="train" if train else "val", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                logits = model(images)
                bce = bce_loss_fn(logits, masks)
                dloss = dice_loss_with_logits(logits, masks)
                loss = bce_weight * bce + dice_weight * dloss

            if train:
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


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_dataloaders(cfg)

    model = FrozenBackboneWithSegHead(cfg.model_dir).to(device)
    optimizer = torch.optim.AdamW(model.seg_head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")

    best_val_dice = -1.0
    history_path = output_dir / "history.csv"
    if not history_path.exists():
        history_path.write_text("epoch,train_loss,train_dice,val_loss,val_dice\n")

    print("Training config:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")
    print(f"  device: {device}")

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_dice = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            train=True,
            bce_weight=cfg.bce_weight,
            dice_weight=cfg.dice_weight,
            amp=cfg.amp,
        )

        val_loss, val_dice = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            train=False,
            bce_weight=cfg.bce_weight,
            dice_weight=cfg.dice_weight,
            amp=cfg.amp,
        )

        print(
            f"Epoch {epoch:03d}/{cfg.epochs:03d} | "
            f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} | "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f}"
        )

        with history_path.open("a") as f:
            f.write(f"{epoch},{train_loss:.6f},{train_dice:.6f},{val_loss:.6f},{val_dice:.6f}\n")

        if epoch % cfg.save_every == 0:
            save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, best_val_dice)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, best_val_dice)
            print(f"  New best checkpoint at epoch {epoch} (val_dice={val_dice:.4f})")

    print(f"Done. Best val Dice: {best_val_dice:.4f}")
