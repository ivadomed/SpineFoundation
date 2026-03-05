"""
Classification trainer — mirrors curia/trainer.py exactly, adapted for local data.

Instead of load_dataset("raidium/CuriaBench", ...) we build a DatasetDict from a
local directory tree.  Everything else (Classifier, SGD, LR scaling, HF Trainer,
feature caching, metrics) is a direct copy of the curia pipeline.
"""

import csv
import fcntl
import os
import sys
import warnings
from datetime import datetime
from functools import partial
from pathlib import Path

import torch.distributed as dist

# Suppress noisy but harmless warnings
warnings.filterwarnings("ignore", message="mtime may not be reliable on this filesystem")
warnings.filterwarnings("ignore", message="find_unused_parameters=True was specified")
warnings.filterwarnings("ignore", message="barrier\\(\\): using the device under current context")
warnings.filterwarnings("ignore", message=".*use_fast.*is set to.*but the image processor class does not have a fast version")

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe on headless servers
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoImageProcessor, AutoModelForImageClassification, Dinov2Model, Trainer, TrainingArguments
from transformers.trainer_callback import PrinterCallback, ProgressCallback


class _QuietProgressCallback(ProgressCallback):
    """Like ProgressCallback but never prints any log dicts (train or eval).
    The tqdm progress bar (epoch/step counter) is kept."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        return  # swallow all dict prints; tqdm bar still updates via on_step_end etc.

from .dataset import (
    build_patch_token_datasets,
    extract_features_fn,
    load_local_dataset,
    preprocess_function,
)
from .model import Classifier


# ── Metrics (verbatim from curia/trainer.py) ─────────────────────────────────


def _extract_predictions_and_labels(eval_pred):
    if hasattr(eval_pred, "predictions"):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    return predictions, labels


def compute_classification_metrics(eval_pred):
    logits, labels = _extract_predictions_and_labels(eval_pred)
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)

    proba = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    if proba.shape[1] == 2:
        proba = proba[:, 1]

    try:
        auc_macro    = roc_auc_score(labels, proba, multi_class="ovr", average="macro")
        auc_weighted = roc_auc_score(labels, proba, multi_class="ovr", average="weighted")
    except ValueError:
        auc_macro = auc_weighted = float("nan")

    return {"accuracy": acc, "auc_ovr_macro": auc_macro, "auc_ovr_weighted": auc_weighted}


# ── CSV result logging ─────────────────────────────────────────────────────────

_TRAIN_CSV_COLUMNS = ["timestamp", "task", "dilation_radius", "auc_ovr_macro", "auc_ovr_weighted", "accuracy"]


def _merge_into_train_csv(csv_path: Path, row: dict) -> None:
    lock_path = csv_path.with_suffix(".lock")
    with open(lock_path, "w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        existing_keys = set()
        rows = []
        if csv_path.exists():
            for r in csv.DictReader(csv_path.open(encoding="utf-8")):
                existing_keys.add((r["timestamp"], r["task"], r["dilation_radius"]))
                rows.append(r)
        key = (row["timestamp"], row["task"], str(row["dilation_radius"]))
        if key not in existing_keys:
            rows.append(row)
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=_TRAIN_CSV_COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)


def _annotate_best(ax, steps, values, mode: str, color: str) -> None:
    """Mark the best point (max or min) on an axis with a dot + label."""
    if not steps or not values:
        return
    arr = np.array(values, dtype=float)
    idx = int(np.nanargmax(arr)) if mode == "max" else int(np.nanargmin(arr))
    bx, by = steps[idx], arr[idx]
    ax.axvline(bx, color=color, linestyle="--", alpha=0.4, linewidth=1)
    ax.plot(bx, by, "o", color=color, markersize=7, zorder=5)
    ax.annotate(
        f"best {by:.4f}\n@step {bx}",
        xy=(bx, by),
        xytext=(8, -14),
        textcoords="offset points",
        fontsize=7.5,
        color=color,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
    )


def _plot_training_history(log_history: list, output_dir: str, task: str) -> None:
    """Parse HF Trainer log_history and save training curves to output_dir."""
    train_steps, train_loss = [], []
    eval_steps, eval_loss   = [], []
    eval_acc, eval_auc_macro, eval_auc_weighted = [], [], []

    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry["step"])
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry["step"])
            eval_loss.append(entry["eval_loss"])
            eval_acc.append(entry.get("eval_accuracy", float("nan")))
            eval_auc_macro.append(entry.get("eval_auc_ovr_macro", float("nan")))
            eval_auc_weighted.append(entry.get("eval_auc_ovr_weighted", float("nan")))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Training curves — {task}", fontsize=13)

    # ── Loss (train + val on same plot to visualise overfitting) ──────────────
    ax = axes[0, 0]
    if train_steps:
        ax.plot(train_steps, train_loss, label="train loss", color="steelblue", alpha=0.6, linewidth=1.2)
    if eval_steps:
        ax.plot(eval_steps, eval_loss, label="val loss", color="orange", linewidth=2)
        _annotate_best(ax, eval_steps, eval_loss, "min", "orange")
    ax.set_title("Loss  (divergence train↗val = overfitting)")
    ax.set_xlabel("step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Accuracy ──────────────────────────────────────────────────────────────
    ax = axes[0, 1]
    if eval_steps:
        ax.plot(eval_steps, eval_acc, color="green", linewidth=2)
        _annotate_best(ax, eval_steps, eval_acc, "max", "green")
    ax.set_title("Val Accuracy")
    ax.set_xlabel("step")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # ── AUC OvR macro ─────────────────────────────────────────────────────────
    ax = axes[1, 0]
    if eval_steps:
        ax.plot(eval_steps, eval_auc_macro, color="crimson", linewidth=2)
        _annotate_best(ax, eval_steps, eval_auc_macro, "max", "crimson")
    ax.set_title("Val AUC OvR macro")
    ax.set_xlabel("step")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # ── AUC OvR weighted ──────────────────────────────────────────────────────
    ax = axes[1, 1]
    if eval_steps:
        ax.plot(eval_steps, eval_auc_weighted, color="purple", linewidth=2)
        _annotate_best(ax, eval_steps, eval_auc_weighted, "max", "purple")
    ax.set_title("Val AUC OvR weighted")
    ax.set_xlabel("step")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(output_dir) / "training_curves.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {out_path}")


def _run_final_eval(trainer, val_dataset, config, timestamp: str, run_dir: str) -> None:
    """Run final evaluation, print detailed report, write to separate logs.
    Also saves val_predictions.npz to run_dir for downstream bootstrap analysis."""
    log_dir = Path(OmegaConf.select(config, "log_dir", default="classification_hf/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    task           = OmegaConf.select(config, "task", default="unknown")
    dilation_radius = int(OmegaConf.select(config, "dilation_radius", default=8))

    # Use a plain inference loop — trainer.predict() calls all_gather (DDP)
    # which crashes after destroy_process_group().
    device = next(trainer.model.parameters()).device
    trainer.model.eval()
    all_logits, all_labels = [], []
    loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=trainer.args.per_device_eval_batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            lbls         = batch["labels"]
            logits_b     = trainer.model(pixel_values)["logits"]
            all_logits.append(logits_b.cpu())
            all_labels.append(lbls)
    logits = torch.cat(all_logits).float().numpy()
    labels = torch.cat(all_labels).numpy()
    proba  = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds  = np.argmax(logits, axis=-1)

    # Save raw predictions for bootstrap analysis
    pred_path = Path(run_dir) / "val_predictions.npz"
    np.savez_compressed(str(pred_path), logits=logits, labels=labels)
    print(f"Predictions saved to {pred_path}")

    acc          = accuracy_score(labels, preds)
    auc_macro    = roc_auc_score(labels, proba, multi_class="ovr", average="macro")
    auc_weighted = roc_auc_score(labels, proba, multi_class="ovr", average="weighted")

    report = classification_report(
        labels, preds,
        target_names=["0 Normal/Mild", "1 Moderate", "2 Severe"],
        digits=4,
        zero_division=0,
    )

    print(f"\n{'='*60}")
    print(f"Task: {task}  |  Dilation: {dilation_radius}  |  {timestamp}")
    print(f"Accuracy       : {acc:.4f}")
    print(f"AUC OvR macro  : {auc_macro:.4f}")
    print(f"AUC OvR weighted: {auc_weighted:.4f}")
    print(report)
    print("="*60)

    # Text log
    log_path = log_dir / f"{task}__dil{dilation_radius}__{timestamp.replace(' ', 'T').replace(':', '')}.log"
    with log_path.open("w") as f:
        f.write(f"Timestamp      : {timestamp}\n")
        f.write(f"Task           : {task}\n")
        f.write(f"Dilation       : {dilation_radius}\n")
        f.write(f"Accuracy       : {acc:.4f}\n")
        f.write(f"AUC OvR macro  : {auc_macro:.4f}\n")
        f.write(f"AUC OvR weighted: {auc_weighted:.4f}\n\n")
        f.write(report)

    # CSV
    csv_path = log_dir / "results.csv"
    row = {
        "timestamp":       timestamp,
        "task":            task,
        "dilation_radius": dilation_radius,
        "auc_ovr_macro":   round(auc_macro, 6),
        "auc_ovr_weighted": round(auc_weighted, 6),
        "accuracy":        round(acc, 6),
    }
    _merge_into_train_csv(csv_path, row)
    print(f"Results written to {csv_path}")


# ── LR scaling (verbatim from curia/trainer.py) ───────────────────────────────


def scale_lr(learning_rate: float, batch_size: int) -> float:
    return learning_rate * batch_size / 256.0


# ── Save head (verbatim from curia/trainer.py) ────────────────────────────────


def save_head(model, output_dir: str):
    output_path = Path(output_dir) / "head.pt"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    payload = {"classifier": model.linear.state_dict()}
    if model.attention_module is not None:
        payload["attention"] = model.attention_module.state_dict()
    torch.save(payload, output_path)
    print(f"Saved classifier head to {output_path}")


# ── instantiate_cache_model_and_dataset (adapted from curia) ─────────────────


def _npz_has_cached(path: str) -> bool:
    """Return True if the NPZ file already contains pre-cached tokens or features."""
    try:
        if Path(path).suffix.lower() != ".npz":
            return False
        files = np.load(path).files
        return "patch_tokens" in files or "features" in files
    except Exception:
        return False


def instantiate_cache_model_and_dataset(config, train_dataset, val_dataset):
    """
    Extract masked-avg-pooled backbone features (once), cache them in the HF
    Dataset, then build a Classifier that trains on those cached features.

    Fast path — features pre-cached in NPZ (via cache_features_to_npz.py):
        The backbone is not loaded at all; features are read directly from disk.
        This saves ~2 GB of GPU memory and the time needed for a full extraction
        pass.

    Slow path — features not yet cached:
        The frozen backbone is run over the full dataset; results are stored in
        the HF Dataset (in-memory, lost on restart).
    """
    model_name      = config.model.model_name
    # processor_name allows using a different model's preprocessor (e.g. curia)
    # when the backbone checkpoint has no preprocessor_config.json
    processor_name  = OmegaConf.select(config, "model.processor_name", default=None) or model_name
    dilation_radius = int(OmegaConf.select(config, "dilation_radius", default=8))
    processor       = AutoImageProcessor.from_pretrained(processor_name, trust_remote_code=True)

    # force_recompute=true bypasses NPZ cache and always runs the backbone
    # (useful when switching to a new encoder whose features aren't cached yet)
    force_recompute = bool(OmegaConf.select(config, "force_recompute", default=False))

    # Peek at the first sample to decide whether to load the backbone
    first_path    = train_dataset[0]["path"]
    use_npz_cache = _npz_has_cached(first_path) and not force_recompute

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device     = torch.device(f"cuda:{local_rank}")

    d_first = np.load(first_path)
    use_patch_tokens = use_npz_cache and "patch_tokens" in d_first

    if use_patch_tokens:
        print("[cache] patch_tokens detected — using on-the-fly PatchTokenDataset (no Map, no backbone).")
        hidden_size = int(d_first["patch_tokens"].shape[-1])
        cache_suffix = OmegaConf.select(config, "cache_suffix", default="") or ""
        train_pt, val_pt = build_patch_token_datasets(
            train_dataset, val_dataset, dilation_radius,
            data_dir=config.data_dir, cache_suffix=cache_suffix,
        )
        attention_cfg = OmegaConf.select(config, "model.attention_cfg")
        model = Classifier(hidden_size, config.model.num_classes, regression=False, attention_cfg=attention_cfg)

        # Load pre-trained classifier weights from curia (same architecture as eval_pretrained.py)
        subfolder = OmegaConf.select(config, "model.subfolder", default=None)
        use_pretrained_head = bool(OmegaConf.select(config, "use_pretrained_head", default=True))
        if subfolder and use_pretrained_head:
            print(f"[pretrained] Loading classifier weights from subfolder='{subfolder}'")
            pretrained = AutoModelForImageClassification.from_pretrained(
                model_name, subfolder=subfolder, trust_remote_code=True
            )
            model.linear.weight.data.copy_(pretrained.classifier.weight.data)
            model.linear.bias.data.copy_(pretrained.classifier.bias.data)
            del pretrained
        else:
            reason = "use_pretrained_head=false" if not use_pretrained_head else "no subfolder"
            print(f"[pretrained] Skipping pretrained head ({reason}) — random init")
            nn.init.normal_(model.linear.weight, mean=0.0, std=0.01)
            nn.init.zeros_(model.linear.bias)

        return model, train_pt, val_pt

    # ── Legacy path: pre-cached pooled 'features' key or no cache → Map ────────
    if use_npz_cache:
        print("[cache] NPZ feature cache detected (pooled) — skipping backbone load.")
        backbone = None
        hidden_size = int(d_first["features"].shape[-1])
    else:
        backbone = Dinov2Model.from_pretrained(model_name, trust_remote_code=True)
        backbone.to(device)
        backbone.eval()
        hidden_size = backbone.config.hidden_size

    _extract = partial(
        extract_features_fn,
        processor=processor,
        backbone=backbone,
        dilation_radius=dilation_radius,
    )

    train_dataset = train_dataset.map(
        _extract, batched=True, batch_size=config.batch_size, num_proc=0
    )
    val_dataset = val_dataset.map(
        _extract, batched=True, batch_size=config.batch_size, num_proc=0
    )

    attention_cfg = OmegaConf.select(config, "model.attention_cfg")
    model = Classifier(
        hidden_size,
        config.model.num_classes,
        regression=False,
        attention_cfg=attention_cfg,
    )
    nn.init.normal_(model.linear.weight, mean=0.0, std=0.01)
    nn.init.zeros_(model.linear.bias)

    for split in [train_dataset, val_dataset]:
        split.set_format(type="torch", columns=["pixel_values", "labels"])

    return model, train_dataset, val_dataset


# ── instantiate_model_and_dataset (full backbone, no caching) ────────────────



def instantiate_model_and_dataset(config, train_dataset, val_dataset):
    """
    Pre-process images with the curia AutoImageProcessor and fine-tune with the
    full backbone frozen — no feature caching.

    If config.model.subfolder is set, it is forwarded to from_pretrained so that
    the correct task head (e.g. "spinal_canal_stenosis") is loaded.  If the
    pre-processed dataset contains a "mask" column (from NPZ files), it is kept
    in the tensor format so the HF Trainer passes it to model(…, mask=…).
    """
    from transformers import AutoModelForImageClassification

    model_name = config.model.model_name
    subfolder  = OmegaConf.select(config, "model.subfolder", default=None)
    processor  = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)

    _preprocess = partial(preprocess_function, processor=processor)

    train_dataset = train_dataset.map(
        _preprocess, batched=True, batch_size=config.batch_size,
        num_proc=config.num_workers,
    )
    val_dataset = val_dataset.map(
        _preprocess, batched=True, batch_size=config.batch_size,
        num_proc=config.num_workers,
    )

    load_kwargs = dict(
        num_labels=config.model.num_classes,
        ignore_mismatched_sizes=True,
        trust_remote_code=True,
    )
    if subfolder:
        load_kwargs["subfolder"] = subfolder

    model = AutoModelForImageClassification.from_pretrained(model_name, **load_kwargs)
    model.base_model.requires_grad_(False)

    columns = ["pixel_values", "labels"]
    if "mask" in train_dataset.column_names:
        columns.append("mask")

    for split in [train_dataset, val_dataset]:
        split.set_format(type="torch", columns=columns)

    return model, train_dataset, val_dataset


# ── Main entry point (mirrors curia/trainer.py main()) ───────────────────────


def main(config) -> None:
    # Silence stdout on non-rank-0 processes — avoids duplicate print lines with DDP
    if int(os.environ.get("RANK", 0)) != 0:
        sys.stdout = open(os.devnull, "w")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Each run gets its own subdirectory so previous runs are never overwritten
    task = OmegaConf.select(config, "task", default="run")
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = str(Path(config.output_dir) / f"{task}__{run_tag}")
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # ── Load local dataset ────────────────────────────────────────────────────
    ds = load_local_dataset(
        config.data_dir,
        val_split=float(OmegaConf.select(config, "val_split", default=0.15)),
        seed=int(OmegaConf.select(config, "seed", default=42)),
    )
    train_dataset = ds["train"]
    val_dataset   = ds["val"]

    print(f"Dataset loaded from {config.data_dir}")
    print(f"  train: {len(train_dataset)}  val: {len(val_dataset)}")

    # ── Build model + preprocessed datasets ───────────────────────────────────
    use_feature_caching = bool(OmegaConf.select(config, "use_feature_caching", default=False))
    if use_feature_caching:
        model, train_dataset, val_dataset = instantiate_cache_model_and_dataset(
            config, train_dataset, val_dataset
        )
    else:
        model, train_dataset, val_dataset = instantiate_model_and_dataset(
            config, train_dataset, val_dataset
        )

    # ── Optimizer + scheduler (verbatim from curia) ───────────────────────────
    steps_per_epoch = max(1, len(train_dataset) // config.batch_size)
    max_steps = steps_per_epoch * config.epochs

    scaled_lr = scale_lr(config.learning_rate, config.batch_size)
    print(f"LR: {config.learning_rate} → scaled: {scaled_lr:.6f}  (batch_size={config.batch_size})")

    weight_decay = float(OmegaConf.select(config, "weight_decay", default=1e-4))
    optimizer = SGD(model.parameters(), lr=scaled_lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=0)
    print(f"Weight decay: {weight_decay}")

    # ── HF TrainingArguments ───────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        logging_strategy="steps",
        logging_steps=max(10, steps_per_epoch // 10),
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,          # restore best val AUC checkpoint
        metric_for_best_model="auc_ovr_macro",
        greater_is_better=True,
        dataloader_num_workers=config.num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=False,
        eval_accumulation_steps=1,   # flush predictions immediately, avoids GPU↔CPU hangs
        ddp_find_unused_parameters=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_classification_metrics,
        optimizers=(optimizer, scheduler),
    )
    trainer.remove_callback(PrinterCallback)   # silence per-step eval metric dicts
    trainer.remove_callback(ProgressCallback)  # replace with quiet version
    trainer.add_callback(_QuietProgressCallback())

    trainer.train()
    trainer.save_model(run_dir)
    save_head(trainer.model, run_dir)

    # Destroy DDP process group HERE so all processes are released together
    # before rank 0 calls predict() — avoids collective-op deadlock.
    # Non-rank-0 processes exit naturally after this point.
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

    # Only rank 0 does final eval + logging (no DDP context anymore)
    if int(os.environ.get("RANK", 0)) == 0:
        _plot_training_history(trainer.state.log_history, run_dir, task)
        _run_final_eval(trainer, val_dataset, config, timestamp, run_dir)

    if sys.stdout != sys.__stdout__:
        sys.stdout.close()
        sys.stdout = sys.__stdout__
    sys.stdout.flush()
    sys.stderr.flush()
