import importlib
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainConfig
from .dataset import FeatureDataset
from .model import ClassificationHead

try:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
    )
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("[warn] scikit-learn not found — AUC and per-class metrics will be unavailable.")


# ── Reproducibility ─────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Checkpoint ──────────────────────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_auc: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_auc": best_val_auc,
        },
        path,
    )


# ── W&B safe import (same pattern as segmentation_hf) ──────────────────────────

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
    raise RuntimeError("Imported 'wandb' module has no 'init'.")


# ── Training epoch ──────────────────────────────────────────────────────────────

def run_train_epoch(
    model: ClassificationHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for features, labels in pbar:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item()
        num_batches += 1

        pbar.set_postfix(
            loss=f"{running_loss / num_batches:.4f}",
            acc=f"{correct / total:.4f}",
        )

    if num_batches == 0:
        return 0.0, 0.0
    return running_loss / num_batches, correct / max(1, total)


# ── Validation epoch ────────────────────────────────────────────────────────────

@torch.no_grad()
def run_val_epoch(
    model: ClassificationHead,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    class_names: List[str],
) -> Dict:
    model.eval()
    running_loss = 0.0
    num_batches = 0
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []

    for features, labels in tqdm(loader, desc="val  ", leave=False):
        features = features.to(device, non_blocking=True)
        labels_dev = labels.to(device, non_blocking=True)

        logits = model(features)
        loss = criterion(logits, labels_dev)
        running_loss += loss.item()
        num_batches += 1

        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_labels.append(labels.numpy())
        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_probs = np.concatenate(all_probs)
    y_pred = np.concatenate(all_preds)
    n_classes = len(class_names)

    results: Dict = {
        "val_loss": running_loss / max(1, num_batches),
        "val_acc": float((y_pred == y_true).mean()),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_probs": y_probs,
    }

    if not SKLEARN_OK:
        return results

    # ── AUC ────────────────────────────────────────────────────────────────────
    if len(np.unique(y_true)) > 1:
        try:
            results["val_auc_macro"] = float(
                roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
            )
            results["val_auc_weighted"] = float(
                roc_auc_score(y_true, y_probs, multi_class="ovr", average="weighted")
            )
            for k, name in enumerate(class_names):
                binary = (y_true == k).astype(int)
                if 0 < binary.sum() < len(binary):
                    results[f"val_auc_class_{name}"] = float(
                        roc_auc_score(binary, y_probs[:, k])
                    )
        except Exception as e:
            print(f"[warn] AUC computation failed: {e}")

    # ── Per-class TP / FN / FP / TN + precision / recall / F1 ─────────────────
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    results["confusion_matrix"] = cm

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )

    per_class: Dict[str, Dict] = {}
    for k, name in enumerate(class_names):
        tp = int(cm[k, k])
        fn = int(cm[k, :].sum() - cm[k, k])
        fp = int(cm[:, k].sum() - cm[k, k])
        tn = int(cm.sum() - tp - fn - fp)
        per_class[name] = {
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn,
            "precision": float(precision[k]),
            "recall": float(recall[k]),
            "f1": float(f1[k]),
            "support": int(support[k]),
        }
    results["per_class"] = per_class

    return results


# ── Pretty-print validation report ─────────────────────────────────────────────

def print_val_report(results: Dict, epoch: int, total_epochs: int) -> None:
    w = 70
    print(f"\n{'─' * w}")
    print(f"  Epoch {epoch:03d}/{total_epochs:03d}  ·  Validation Report")
    print(f"{'─' * w}")

    line = (
        f"  loss={results['val_loss']:.4f}"
        f"  acc={results['val_acc']:.4f}"
    )
    if "val_auc_macro" in results:
        line += f"  AUC-macro={results['val_auc_macro']:.4f}"
    if "val_auc_weighted" in results:
        line += f"  AUC-weighted={results['val_auc_weighted']:.4f}"
    print(line)

    if "per_class" in results:
        hdr = f"  {'Class':>8}  {'Support':>8}  {'TP':>6}  {'FN':>6}  {'FP':>6}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'AUC':>7}"
        print(f"\n{hdr}")
        print(f"  {'─' * (w - 2)}")
        for name, s in results["per_class"].items():
            auc_key = f"val_auc_class_{name}"
            auc_str = f"{results[auc_key]:.4f}" if auc_key in results else "   N/A"
            print(
                f"  {name:>8}  {s['support']:>8}  {s['TP']:>6}  {s['FN']:>6}  {s['FP']:>6}"
                f"  {s['precision']:>7.4f}  {s['recall']:>7.4f}  {s['f1']:>7.4f}  {auc_str:>7}"
            )

    if "confusion_matrix" in results:
        cm = results["confusion_matrix"]
        print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
        for row in cm:
            print("    " + "  ".join(f"{v:>6}" for v in row))

    print(f"{'─' * w}\n")


# ── Main training loop ──────────────────────────────────────────────────────────

def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Datasets ──────────────────────────────────────────────────────────────
    features_dir = Path(cfg.features_dir)
    train_ds = FeatureDataset(str(features_dir / "train.npz"))
    val_ds   = FeatureDataset(str(features_dir / "val.npz"))
    class_names = train_ds.class_names

    print(f"Train: {len(train_ds)} samples  feature_dim={train_ds.feature_dim}  num_classes={train_ds.num_classes}")
    print(f"Val  : {len(val_ds)} samples")
    print(f"Class names: {class_names}")
    counts = train_ds.class_counts
    for name, cnt in zip(class_names, counts):
        print(f"  class {name:>4s}: {cnt:>6d}  ({100*cnt/len(train_ds):.1f}%)")

    # ── Class-weighted loss ───────────────────────────────────────────────────
    if cfg.use_class_weights:
        raw = counts.astype(np.float32)
        weights = 1.0 / np.maximum(raw, 1)
        weights = weights / weights.sum() * len(weights)
        class_weights = torch.from_numpy(weights).float().to(device)
        print(f"Class weights: {weights.round(3)}")
    else:
        class_weights = None
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ClassificationHead(
        in_features=train_ds.feature_dim,
        num_classes=train_ds.num_classes,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # ── W&B ───────────────────────────────────────────────────────────────────
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

    # ── History ───────────────────────────────────────────────────────────────
    best_val_auc = -1.0
    history_path = output_dir / "history.csv"
    if not history_path.exists():
        history_path.write_text("epoch,train_loss,train_acc,val_loss,val_acc,val_auc_macro\n")

    print(f"\nTraining on {device}  —  {cfg.epochs} epochs")
    print(f"{'─' * 70}")

    try:
        for epoch in range(1, cfg.epochs + 1):
            train_loss, train_acc = run_train_epoch(model, train_loader, optimizer, device, criterion)
            val_results = run_val_epoch(model, val_loader, device, criterion, class_names)
            scheduler.step()

            val_auc = val_results.get("val_auc_macro", 0.0)

            print(
                f"Epoch {epoch:03d}/{cfg.epochs:03d} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_results['val_loss']:.4f} val_acc={val_results['val_acc']:.4f} "
                f"AUC={val_auc:.4f}"
            )

            print_val_report(val_results, epoch, cfg.epochs)

            # ── CSV ───────────────────────────────────────────────────────────
            with history_path.open("a") as f:
                f.write(
                    f"{epoch},{train_loss:.6f},{train_acc:.6f},"
                    f"{val_results['val_loss']:.6f},{val_results['val_acc']:.6f},{val_auc:.6f}\n"
                )

            # ── W&B log ───────────────────────────────────────────────────────
            if wandb_run is not None:
                payload: Dict = {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "val/loss": val_results["val_loss"],
                    "val/acc": val_results["val_acc"],
                    "val/auc_macro": val_auc,
                    "val/auc_weighted": val_results.get("val_auc_weighted", 0.0),
                    "best/val_auc": max(best_val_auc, val_auc),
                    "lr": scheduler.get_last_lr()[0],
                }
                if "per_class" in val_results:
                    for name, s in val_results["per_class"].items():
                        payload[f"val/{name}/TP"]        = s["TP"]
                        payload[f"val/{name}/FN"]        = s["FN"]
                        payload[f"val/{name}/FP"]        = s["FP"]
                        payload[f"val/{name}/precision"] = s["precision"]
                        payload[f"val/{name}/recall"]    = s["recall"]
                        payload[f"val/{name}/f1"]        = s["f1"]
                    for name in class_names:
                        auc_key = f"val_auc_class_{name}"
                        if auc_key in val_results:
                            payload[f"val/{name}/auc"] = val_results[auc_key]
                wandb_run.log(payload)

            # ── Checkpoints ───────────────────────────────────────────────────
            save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, best_val_auc)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, best_val_auc)
                print(f"  ★ New best AUC: {best_val_auc:.4f}")

    finally:
        if wandb_run is not None:
            wandb_run.finish()

    print(f"\nDone. Best val AUC-macro: {best_val_auc:.4f}")
