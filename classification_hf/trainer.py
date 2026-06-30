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
from tqdm import tqdm as _tqdm
from omegaconf import OmegaConf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
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
    CropTokenDataset,
    build_patch_token_datasets,
    extract_features_fn,
    load_fold_dataset,
    load_local_dataset,
    load_test_dataset,
    preprocess_function,
)
from .model import Classifier, MaskedBackboneClassifier, TokenGridClassifier


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


_WBCE_WEIGHTS = np.array([1.0, 2.0, 4.0])

def _wbce(logits_np: np.ndarray, labels_np: np.ndarray) -> float:
    """Weighted cross-entropy with challenge weights [1, 2, 4]."""
    from scipy.special import softmax as sp_softmax
    probs = np.clip(sp_softmax(logits_np.astype(np.float64), axis=1), 1e-7, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    return float(np.mean(_WBCE_WEIGHTS[labels_np] * (-np.log(probs[np.arange(len(labels_np)), labels_np]))))


def compute_classification_metrics(eval_pred):
    logits, labels = _extract_predictions_and_labels(eval_pred)
    preds = np.argmax(logits, axis=-1)

    proba = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    proba_for_auc = proba[:, 1] if proba.shape[1] == 2 else proba

    try:
        auc_macro    = roc_auc_score(labels, proba_for_auc, multi_class="ovr", average="macro")
        auc_weighted = roc_auc_score(labels, proba_for_auc, multi_class="ovr", average="weighted")
    except ValueError:
        auc_macro = auc_weighted = float("nan")

    return {
        "accuracy":          accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "f1_macro":          f1_score(labels, preds, average="macro",    zero_division=0),
        "f1_weighted":       f1_score(labels, preds, average="weighted", zero_division=0),
        "qwk":               cohen_kappa_score(labels, preds, weights="quadratic"),
        "kappa":             cohen_kappa_score(labels, preds),
        "mcc":               matthews_corrcoef(labels, preds),
        "auc_ovr_macro":     auc_macro,
        "auc_ovr_weighted":  auc_weighted,
        "wbce":              _wbce(logits, labels),
    }


# ── CSV result logging ─────────────────────────────────────────────────────────

_TRAIN_CSV_COLUMNS = [
    "timestamp", "task",
    "auc_ovr_macro", "auc_ovr_weighted",
    "f1_macro", "f1_weighted",
    "qwk", "kappa", "mcc",
    "accuracy", "balanced_accuracy",
]


def _merge_into_train_csv(csv_path: Path, row: dict) -> None:
    lock_path = csv_path.with_suffix(".lock")
    with open(lock_path, "w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        existing_keys = set()
        rows = []
        if csv_path.exists():
            for r in csv.DictReader(csv_path.open(encoding="utf-8")):
                existing_keys.add((r["timestamp"], r["task"]))
                rows.append(r)
        key = (row["timestamp"], row["task"])
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


def _run_final_eval(model, val_dataset, eval_batch_size: int, config, timestamp: str, run_dir: str,
                    split_name: str = "val", sample_paths: list = None) -> None:
    """Run final evaluation, print detailed report, write to separate logs.
    Saves {split_name}_predictions.npz to run_dir for downstream bootstrap analysis."""
    log_dir = Path(OmegaConf.select(config, "log_dir", default="classification_hf/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    task = OmegaConf.select(config, "task", default="unknown")

    # HF Trainer may alter the dataset format in-place during training; restore it.
    if hasattr(val_dataset, "set_format") and hasattr(val_dataset, "column_names"):
        cols = ["pixel_values", "labels"]
        if "mask" in val_dataset.column_names:
            cols.append("mask")
        val_dataset.set_format(type="torch", columns=cols)

    device = next(model.parameters()).device
    model.eval()
    all_logits, all_labels = [], []
    loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=eval_batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device).float()
            lbls         = batch["labels"]
            mask_b   = batch["mask"].to(device) if "mask" in batch else None
            logits_b = (model(pixel_values, mask=mask_b) if mask_b is not None
                        else model(pixel_values))["logits"]
            all_logits.append(logits_b.cpu())
            all_labels.append(lbls)
    logits = torch.cat(all_logits).float().numpy()
    labels = torch.cat(all_labels).numpy()
    proba  = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds  = np.argmax(logits, axis=-1)

    # Save raw predictions for bootstrap analysis
    pred_path = Path(run_dir) / f"{split_name}_predictions.npz"
    sample_names = np.array([Path(p).stem for p in sample_paths]) if sample_paths is not None else np.array([])
    np.savez_compressed(str(pred_path), logits=logits, labels=labels, sample_names=sample_names)
    print(f"Predictions saved to {pred_path}")

    try:
        auc_macro    = roc_auc_score(labels, proba, multi_class="ovr", average="macro")
        auc_weighted = roc_auc_score(labels, proba, multi_class="ovr", average="weighted")
        n_classes    = proba.shape[1]
        auc_per_class = [
            roc_auc_score((labels == c).astype(int), proba[:, c])
            for c in range(n_classes)
        ]
    except ValueError:
        auc_macro = auc_weighted = float("nan")
        auc_per_class = [float("nan")] * proba.shape[1]

    acc               = accuracy_score(labels, preds)
    bal_acc           = balanced_accuracy_score(labels, preds)
    f1_mac            = f1_score(labels, preds, average="macro",    zero_division=0)
    f1_wgt            = f1_score(labels, preds, average="weighted", zero_division=0)
    qwk               = cohen_kappa_score(labels, preds, weights="quadratic")
    kappa             = cohen_kappa_score(labels, preds)
    mcc               = matthews_corrcoef(labels, preds)

    all_names = ["0 Normal/Mild", "1 Moderate", "2 Severe"]
    present   = sorted(np.unique(labels).tolist())
    report = classification_report(
        labels, preds,
        labels=present,
        target_names=[all_names[i] for i in present],
        digits=4,
        zero_division=0,
    )

    print(f"\n{'='*60}")
    print(f"Task: {task}  |  {timestamp}")
    print(f"Accuracy          : {acc:.4f}")
    print(f"Balanced accuracy : {bal_acc:.4f}")
    print(f"F1 macro          : {f1_mac:.4f}")
    print(f"F1 weighted       : {f1_wgt:.4f}")
    print(f"QWK               : {qwk:.4f}")
    print(f"Kappa             : {kappa:.4f}")
    print(f"MCC               : {mcc:.4f}")
    print(f"AUC OvR macro     : {auc_macro:.4f}")
    print(f"AUC OvR weighted  : {auc_weighted:.4f}")
    auc_names = ["Normal/Mild", "Moderate", "Severe"]
    for c, a in enumerate(auc_per_class):
        print(f"  AUC {auc_names[c]:<12}: {a:.4f}")
    print(report)
    print("="*60)

    # Text log
    log_path = log_dir / f"{task}__{timestamp.replace(' ', 'T').replace(':', '')}.log"
    with log_path.open("w") as f:
        f.write(f"Timestamp         : {timestamp}\n")
        f.write(f"Task              : {task}\n")
        f.write(f"Accuracy          : {acc:.4f}\n")
        f.write(f"Balanced accuracy : {bal_acc:.4f}\n")
        f.write(f"F1 macro          : {f1_mac:.4f}\n")
        f.write(f"F1 weighted       : {f1_wgt:.4f}\n")
        f.write(f"QWK               : {qwk:.4f}\n")
        f.write(f"Kappa             : {kappa:.4f}\n")
        f.write(f"MCC               : {mcc:.4f}\n")
        f.write(f"AUC OvR macro     : {auc_macro:.4f}\n")
        f.write(f"AUC OvR weighted  : {auc_weighted:.4f}\n")
        for c, a in enumerate(auc_per_class):
            f.write(f"AUC {auc_names[c]:<12} : {a:.4f}\n")
        f.write(f"\n{report}")

    # CSV
    csv_path = log_dir / "results.csv"
    row = {
        "timestamp":        timestamp,
        "task":             task,
        "auc_ovr_macro":    round(auc_macro,    6),
        "auc_ovr_weighted": round(auc_weighted, 6),
        "f1_macro":         round(f1_mac,        6),
        "f1_weighted":      round(f1_wgt,        6),
        "qwk":              round(qwk,           6),
        "kappa":            round(kappa,         6),
        "mcc":              round(mcc,           6),
        "accuracy":         round(acc,           6),
        "balanced_accuracy": round(bal_acc,      6),
    }
    _merge_into_train_csv(csv_path, row)
    print(f"Results written to {csv_path}")


# ── Test-set inference (fold split only) ─────────────────────────────────────


def _run_test_eval(model, config, fold_split_csv: str, run_dir: str, timestamp: str,
                   pt_cache=None) -> None:
    """
    Run inference on all is_test=1 subjects and save test_predictions.npz to run_dir.

    Requires the pooled-features cache (~/.cache/classification_hf/pooled_features_*_dil{N}.pt),
    since test subjects must go through the same feature extraction as train/val.
    pt_cache: pre-loaded cache dict (avoids reloading from NFS if already in memory).
    """
    from .dataset import _DictDataset

    # ── Build test HF Dataset ─────────────────────────────────────────────────
    hf_test = load_test_dataset(config.data_dir, fold_split_csv)

    # ── Locate pooled-features cache ──────────────────────────────────────────
    cache_suffix = OmegaConf.select(config, "cache_suffix", default="") or ""
    suffix_part  = f"_{cache_suffix}" if cache_suffix else ""
    _cache_root  = Path.home() / ".cache" / "classification_hf"
    pt_path      = _cache_root / f"pooled_features_{Path(config.data_dir).name}{suffix_part}.pt"

    if pt_cache is not None:
        cache = pt_cache
        print(f"[test] Reusing in-memory cache (skipping NFS reload).")
    elif not pt_path.exists():
        print(f"[test] WARNING: cache not found at {pt_path} — skipping test inference")
        return
    else:
        cache = torch.load(pt_path, weights_only=True)
    path_to_idx = {p: i for i, p in enumerate(cache["paths"])}

    # Filter to paths present in the cache (graceful for partial caches)
    test_paths   = hf_test["path"]
    test_targets = hf_test["target"]
    idxs, keep_targets = [], []
    missing = 0
    for p, t in zip(test_paths, test_targets):
        if p in path_to_idx:
            idxs.append(path_to_idx[p])
            keep_targets.append(t)
        else:
            missing += 1
    if missing:
        print(f"[test] WARNING: {missing} test files not found in cache — skipped")
    if not idxs:
        print("[test] No test samples found in cache — skipping test inference")
        return

    feats  = cache["features"][idxs]                          # (N, D)
    labels = torch.tensor(keep_targets, dtype=torch.long)     # (N,)
    test_ds = _DictDataset(torch.utils.data.TensorDataset(feats, labels))

    # ── Inference ─────────────────────────────────────────────────────────────
    _run_final_eval(model, test_ds, int(config.batch_size), config, timestamp, run_dir,
                    split_name="test", sample_paths=test_paths)


# ── LR scaling (verbatim from curia/trainer.py) ───────────────────────────────


def scale_lr(learning_rate: float, batch_size: int) -> float:
    return learning_rate * batch_size / 256.0


# ── Save head (verbatim from curia/trainer.py) ────────────────────────────────


def save_head(model, output_dir: str):
    output_path = Path(output_dir) / "head.pt"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    unwrapped = getattr(model, "_orig_mod", model)
    if isinstance(unwrapped, TokenGridClassifier):
        torch.save({"state_dict": unwrapped.state_dict()}, output_path)
    elif isinstance(unwrapped, (Classifier, MaskedBackboneClassifier)):
        payload = {"classifier": unwrapped.linear.state_dict()}
        attn = getattr(unwrapped, "attention_module", None)
        if attn is not None:
            payload["attention"] = attn.state_dict()
        torch.save(payload, output_path)
    else:
        # Full HF model (e.g. Dinov2ForImageClassification) — already saved by trainer.save_model()
        print(f"[save_head] Full HF model — skipping head.pt (already saved by Trainer)")
        return
    print(f"Saved classifier head to {output_path}")


# ── instantiate_cache_model_and_dataset (adapted from curia) ─────────────────


def _npz_has_cached(path: str, token_key: str = "patch_tokens") -> bool:
    """Return True if the NPZ file already contains pre-cached tokens or features."""
    try:
        if Path(path).suffix.lower() != ".npz":
            return False
        files = np.load(path).files
        return token_key in files or "features" in files
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
    processor = AutoImageProcessor.from_pretrained(processor_name, trust_remote_code=True)

    # force_recompute=true bypasses NPZ cache and always runs the backbone
    force_recompute = bool(OmegaConf.select(config, "force_recompute", default=False))

    # Derive token_key from cache_suffix (e.g. "custom" → "patch_tokens_custom")
    cache_suffix = OmegaConf.select(config, "cache_suffix", default="") or ""
    token_key    = f"patch_tokens_{cache_suffix}" if cache_suffix else "patch_tokens"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device     = torch.device(f"cuda:{local_rank}")

    # Fast path: pooled_features .pt cache exists → skip NPZ inspection entirely
    suffix_part = f"_{cache_suffix}" if cache_suffix else ""
    _cache_root = Path.home() / ".cache" / "classification_hf"
    pt_path = _cache_root / f"pooled_features_{Path(config.data_dir).name}{suffix_part}.pt"
    use_pt_cache = pt_path.exists() and not force_recompute

    # Peek at the first sample to decide whether to load the backbone
    first_path    = train_dataset[0]["path"]
    use_npz_cache = _npz_has_cached(first_path, token_key) and not force_recompute

    d_first = np.load(first_path)
    use_patch_tokens = use_pt_cache or (use_npz_cache and token_key in d_first.files)

    if use_patch_tokens:
        if use_pt_cache:
            print(f"[cache] {pt_path.name} detected — loading pooled features directly (no Map, no backbone).")
            _pt_cache = torch.load(pt_path, weights_only=True)
            hidden_size = int(_pt_cache["features"].shape[1])
            # Expose cache so callers can reuse it for test-set inference (avoids a second NFS load)
            instantiate_cache_model_and_dataset._last_pt_cache = _pt_cache
        else:
            print(f"[cache] {token_key} detected — using on-the-fly PatchTokenDataset (no Map, no backbone).")
            hidden_size = int(d_first[token_key].shape[-1])
        train_pt, val_pt = build_patch_token_datasets(
            train_dataset, val_dataset,
            data_dir=config.data_dir, cache_suffix=cache_suffix, token_key=token_key,
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

    model_name      = config.model.model_name
    subfolder       = OmegaConf.select(config, "model.subfolder", default=None)
    processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)

    crop_cm    = float(OmegaConf.select(config, "crop_cm", default=None) or 0) or None
    _preprocess = partial(preprocess_function, processor=processor, crop_cm=crop_cm)
    if crop_cm:
        print(f"[backbone] crop_cm={crop_cm}")

    train_dataset = train_dataset.map(
        _preprocess, batched=True, batch_size=config.batch_size,
        num_proc=None,
    )
    val_dataset = val_dataset.map(
        _preprocess, batched=True, batch_size=config.batch_size,
        num_proc=None,
    )

    backbone    = Dinov2Model.from_pretrained(model_name, trust_remote_code=True)
    hidden_size = backbone.config.hidden_size
    model       = MaskedBackboneClassifier(backbone, hidden_size, config.model.num_classes)

    use_pretrained_head = bool(OmegaConf.select(config, "use_pretrained_head", default=True))
    if subfolder and use_pretrained_head:
        print(f"[pretrained] Loading classifier weights from subfolder='{subfolder}'")
        pretrained = AutoModelForImageClassification.from_pretrained(
            model_name, subfolder=subfolder, trust_remote_code=True,
            num_labels=config.model.num_classes, ignore_mismatched_sizes=True, low_cpu_mem_usage=False,
        )
        model.linear.weight.data.copy_(pretrained.classifier.weight.data)
        model.linear.bias.data.copy_(pretrained.classifier.bias.data)
        del pretrained
    else:
        reason = "use_pretrained_head=false" if not use_pretrained_head else "no subfolder"
        print(f"[pretrained] Skipping pretrained head ({reason}) — random init")
        nn.init.normal_(model.linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(model.linear.bias)

    freeze_backbone = bool(OmegaConf.select(config, "freeze_backbone", default=True))
    if freeze_backbone:
        model.backbone.requires_grad_(False)
        print("[backbone] frozen")
    else:
        print("[backbone] unfrozen — full fine-tuning")
    if bool(OmegaConf.select(config, "compile_model", default=False)):
        model = torch.compile(model)
        print("[backbone] torch.compile enabled")

    columns = ["pixel_values", "labels"]
    if "mask" in train_dataset.column_names:
        columns.append("mask")

    for split in [train_dataset, val_dataset]:
        split.set_format(type="torch", columns=columns)

    return model, train_dataset, val_dataset


# ── TokenGridClassifier instantiation ────────────────────────────────────────


def instantiate_resnet_model_and_dataset(config, train_hf, val_hf):
    """
    Build a TokenGridClassifier + CropTokenDataset pair.

    Reads the full (un-pooled) patch token grid from each NPZ file at training
    time via a DataLoader.  No backbone is loaded; the tokens must already be
    cached in the NPZ (via cache_patch_tokens.py --crop_cm 4 --suffix <suffix>).

    Config keys read:
        model.num_classes    (default 3)
        model.proj_dim       (default 128)
        model.n_blocks       (default 2)
        cache_suffix         (default "crop4cm") → token_key = patch_tokens_{suffix}
    """
    cache_suffix = OmegaConf.select(config, "cache_suffix", default="crop4cm") or "crop4cm"
    token_key    = f"patch_tokens_{cache_suffix}"
    proj_dim     = int(OmegaConf.select(config, "model.proj_dim",   default=128))
    n_blocks     = int(OmegaConf.select(config, "model.n_blocks",   default=2))
    num_classes  = int(OmegaConf.select(config, "model.num_classes", default=3))

    # Peek at first sample to infer hidden_size (D)
    import numpy as _np
    first_path = train_hf[0]["path"]
    d_first = _np.load(first_path)
    if token_key not in d_first.files:
        raise KeyError(
            f"Token key '{token_key}' not found in {first_path}.\n"
            f"Run cache_patch_tokens.py --suffix {cache_suffix} --crop_cm 4 first."
        )
    hidden_size = int(d_first[token_key].shape[-1])
    print(f"[resnet] token_key={token_key}  hidden_size={hidden_size}  "
          f"proj_dim={proj_dim}  n_blocks={n_blocks}")

    class_weights = None
    if OmegaConf.select(config, "use_class_weights", default=False):
        from sklearn.utils.class_weight import compute_class_weight
        train_labels = _np.array(train_hf["target"])
        classes = _np.arange(num_classes)
        cw = compute_class_weight("balanced", classes=classes, y=train_labels)
        class_weights = torch.tensor(cw, dtype=torch.float32)
        print(f"[resnet] class_weights={cw.round(3)}")

    model = TokenGridClassifier(hidden_size, num_classes, proj_dim=proj_dim, n_blocks=n_blocks,
                                class_weights=class_weights)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[resnet] model params: {n_params:,}")
    train_ds = CropTokenDataset(train_hf["path"], train_hf["target"], token_key, preload=True)
    val_ds   = CropTokenDataset(val_hf["path"],   val_hf["target"],   token_key, preload=True)
    print(f"[resnet] train={len(train_ds)} samples  val={len(val_ds)} samples")
    return model, train_ds, val_ds


# ── Training loop for TokenGridClassifier (DataLoader, not all-on-GPU) ────────


def _train_token_grid_model(model, train_dataset, val_dataset, config,
                            run_dir: str, task: str, timestamp: str,
                            trial=None) -> float:
    """
    Custom training loop for TokenGridClassifier.

    Unlike _fast_train_linear, data is NOT preloaded onto GPU (token grids are
    ~3 MB per sample).  A standard DataLoader with num_workers workers is used.
    Best checkpoint (by val AUC macro) is restored at the end.
    """
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))
    device      = torch.device(f"cuda:{local_rank}")
    model.to(device)
    torch.set_num_threads(4)

    batch_size   = int(config.batch_size)
    epochs       = int(config.epochs)
    num_workers  = int(OmegaConf.select(config, "num_workers", default=8))
    weight_decay = float(OmegaConf.select(config, "weight_decay", default=1e-4))
    scaled_lr    = scale_lr(config.learning_rate, batch_size)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False, persistent_workers=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False, persistent_workers=False,
    )

    steps_per_epoch = len(train_loader)
    max_steps       = steps_per_epoch * epochs

    optimizer = SGD(model.parameters(), lr=scaled_lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=0)

    print(f"LR: {config.learning_rate} → scaled: {scaled_lr:.6f}  (batch_size={batch_size})")
    print(f"TokenGrid train: {len(train_dataset)} samples, {epochs} epochs, "
          f"{steps_per_epoch} steps/epoch, {num_workers} workers")

    # Preload to GPU if data is already in RAM (CropTokenDataset with preload=True)
    # This eliminates DataLoader overhead entirely — pure tensor indexing on GPU
    if hasattr(train_dataset, "tokens") and train_dataset.tokens is not None:
        print("[resnet] Data in RAM → moving to GPU for zero-overhead training...", flush=True)
        train_tokens_gpu = train_dataset.tokens.to(device)   # float16 on GPU
        train_labels_gpu = torch.tensor(train_dataset.labels, device=device, dtype=torch.long)
        val_tokens_gpu   = val_dataset.tokens.to(device)     # float16 on GPU
        val_labels_gpu   = torch.tensor(val_dataset.labels, device=device, dtype=torch.long)
        use_gpu_tensors  = True
        gb = (train_tokens_gpu.numel() + val_tokens_gpu.numel()) * 2 / 1e9
        print(f"[resnet] {gb:.1f} GB float16 on GPU", flush=True)
    else:
        use_gpu_tensors = False

    best_val_loss = float("inf")
    best_state    = None
    log_history   = []
    step          = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        if use_gpu_tensors:
            perm = torch.randperm(len(train_dataset), device=device)
            n_batches = (len(train_dataset) + batch_size - 1) // batch_size
            bar = _tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{epochs} [train]",
                        unit="batch", leave=False)
            for i in bar:
                idx = perm[i * batch_size:(i + 1) * batch_size]
                pv  = train_tokens_gpu[idx].float()
                lbl = train_labels_gpu[idx]
                out = model(pv, lbl)
                optimizer.zero_grad(set_to_none=True)
                out["loss"].backward()
                optimizer.step()
                scheduler.step()
                loss_val = out["loss"].item()
                epoch_loss += loss_val
                step += 1
                bar.set_postfix(loss=f"{loss_val:.4f}")
            steps_this_epoch = n_batches
        else:
            train_bar = _tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]",
                              unit="batch", leave=False)
            for batch in train_bar:
                pv  = batch["pixel_values"].to(device, non_blocking=True).float()
                lbl = batch["labels"].to(device, non_blocking=True)
                out = model(pv, lbl)
                optimizer.zero_grad(set_to_none=True)
                out["loss"].backward()
                optimizer.step()
                scheduler.step()
                loss_val = out["loss"].item()
                epoch_loss += loss_val
                step += 1
                train_bar.set_postfix(loss=f"{loss_val:.4f}")
            steps_this_epoch = steps_per_epoch

        # Validation
        model.eval()
        all_logits, all_labels_list = [], []
        val_loss_sum = 0.0
        with torch.no_grad():
            if use_gpu_tensors:
                n_val_batches = (len(val_dataset) + batch_size - 1) // batch_size
                for i in range(n_val_batches):
                    pv  = val_tokens_gpu[i * batch_size:(i + 1) * batch_size].float()
                    lbl = val_labels_gpu[i * batch_size:(i + 1) * batch_size]
                    out = model(pv, lbl)
                    val_loss_sum += out["loss"].item() * len(lbl)
                    all_logits.append(out["logits"].cpu())
                    all_labels_list.append(lbl.cpu())
            else:
                for batch in _tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]",
                                   unit="batch", leave=False):
                    pv  = batch["pixel_values"].to(device, non_blocking=True).float()
                    lbl = batch["labels"].to(device, non_blocking=True)
                    out = model(pv, lbl)
                    val_loss_sum += out["loss"].item() * len(lbl)
                    all_logits.append(out["logits"].cpu())
                    all_labels_list.append(batch["labels"])
        logits_np = torch.cat(all_logits).float().numpy()
        labels_np = torch.cat(all_labels_list).numpy()
        metrics   = compute_classification_metrics((logits_np, labels_np))

        train_loss = epoch_loss / (steps_this_epoch if use_gpu_tensors else steps_per_epoch)
        val_loss   = _wbce(logits_np, labels_np)
        log_history.append({
            "step": step,
            "loss": train_loss,
            "eval_loss": val_loss,
            "eval_accuracy":         metrics["accuracy"],
            "eval_auc_ovr_macro":    metrics["auc_ovr_macro"],
            "eval_auc_ovr_weighted": metrics["auc_ovr_weighted"],
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if trial is not None:
            import optuna
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"auc_macro={metrics['auc_ovr_macro']:.4f}  "
                  f"acc={metrics['accuracy']:.4f}", flush=True)

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"Best val loss: {best_val_loss:.4f}")

    _plot_training_history(log_history, run_dir, task)
    save_head(model, run_dir)
    _run_final_eval(model, val_dataset, batch_size, config, timestamp, run_dir,
                    sample_paths=val_dataset.paths)
    return best_val_loss


# ── Fast training loop for pre-cached features ───────────────────────────────


def _fast_train_linear(model, train_dataset, val_dataset, config, run_dir: str,
                       task: str, timestamp: str, trial=None) -> float:
    """
    Bypass HF Trainer entirely for the _DictDataset fast path.

    All features and labels are loaded onto the GPU once; training is a plain
    PyTorch loop with no DataLoader overhead.  ~10-50x faster than HF Trainer
    for a linear head on cached features.
    """
    from classification_hf.dataset import _DictDataset as _DD
    assert isinstance(train_dataset, _DD)

    # ── Load all data to GPU in one shot ──────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    model.to(device)

    # _DictDataset wraps TensorDataset — access tensors directly (no Python loop)
    train_x, train_y = train_dataset._ds.tensors   # (N, D), (N,)
    val_x,   val_y   = val_dataset._ds.tensors
    train_x, train_y = train_x.to(device), train_y.to(device)
    # val stays on CPU; moved in chunks during eval

    # ── Optimizer + cosine schedule ───────────────────────────────────────────
    batch_size   = int(config.batch_size)
    epochs       = int(config.epochs)
    scaled_lr    = scale_lr(config.learning_rate, batch_size)
    weight_decay = float(OmegaConf.select(config, "weight_decay", default=1e-4))
    n_train      = train_x.shape[0]
    steps_per_epoch = max(1, n_train // batch_size)
    max_steps       = steps_per_epoch * epochs

    optimizer = SGD(model.parameters(), lr=scaled_lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=0)

    print(f"LR: {config.learning_rate} → scaled: {scaled_lr:.6f}  (batch_size={batch_size})")
    print(f"Fast train: {n_train} samples, {epochs} epochs, {steps_per_epoch} steps/epoch")

    val_x_dev = val_x.to(device)

    # ── Training loop ─────────────────────────────────────────────────────────
    patience      = int(OmegaConf.select(config, "early_stopping_patience", default=30))
    no_improve    = 0
    best_val_loss = float("inf")
    best_state    = None
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            out = model(train_x[idx], train_y[idx])
            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            optimizer.step()
            scheduler.step()

        # Val WBCE every epoch (cheap for linear head)
        model.eval()
        val_logits_list, val_labels_list = [], []
        with torch.no_grad():
            for start in range(0, val_x_dev.shape[0], batch_size):
                vx = val_x_dev[start:start + batch_size]
                vy = val_y[start:start + batch_size].to(device)
                vout = model(vx, vy)
                val_logits_list.append(vout["logits"].cpu())
                val_labels_list.append(vy.cpu())
        val_logits_np = torch.cat(val_logits_list).float().numpy()
        val_labels_np = torch.cat(val_labels_list).numpy()
        val_loss = _wbce(val_logits_np, val_labels_np)
        model.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if trial is not None:
            import optuna
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch+1}/{epochs}  train_loss={out['loss'].item():.4f}  val_loss={val_loss:.4f}", flush=True)

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)", flush=True)
            break

    # ── Save best model + final eval ─────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best val loss: {best_val_loss:.4f}")
    save_head(model, run_dir)
    _run_final_eval(model, val_dataset, batch_size, config, timestamp, run_dir)
    return best_val_loss


# ── Eval-only entry point ─────────────────────────────────────────────────────


def eval_only(config, run_dir: str) -> None:
    """Load head.pt from run_dir and run _run_final_eval without retraining."""
    from datetime import datetime
    run_dir  = str(run_dir)
    head_path = Path(run_dir) / "head.pt"
    if not head_path.exists():
        raise FileNotFoundError(f"No head.pt found in {run_dir}")

    task      = OmegaConf.select(config, "task", default="unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    head_type = OmegaConf.select(config, "model.head_type", default="linear") or "linear"
    if head_type != "resnet":
        raise NotImplementedError("eval_only currently only supports head_type=resnet")

    # Build val dataset only (train is None — CropTokenDataset handles it)
    _, hf_val = _load_fold_split(config)
    _, val_dataset = _eval_only_build_val(config, hf_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)

    # Reconstruct model and load weights
    ckpt = torch.load(head_path, map_location="cpu", weights_only=True)
    num_classes  = int(OmegaConf.select(config, "model.num_classes", default=3))
    proj_dim     = int(OmegaConf.select(config, "model.proj_dim",    default=128))
    n_blocks     = int(OmegaConf.select(config, "model.n_blocks",    default=2))
    hidden_size  = int(ckpt["state_dict"]["proj.0.weight"].shape[1])
    model = TokenGridClassifier(hidden_size, num_classes,
                                proj_dim=proj_dim, n_blocks=n_blocks)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(device).eval()
    print(f"Loaded head.pt from {head_path}")

    batch_size = int(config.batch_size)
    _run_final_eval(model, val_dataset, batch_size, config, timestamp, run_dir,
                    sample_paths=val_dataset.paths)


def _eval_only_build_val(config, hf_val):
    """Build only val dataset for eval_only mode."""
    cache_suffix = OmegaConf.select(config, "cache_suffix", default="crop4cm") or "crop4cm"
    token_key    = f"patch_tokens_{cache_suffix}"
    val_ds = CropTokenDataset(hf_val["path"], hf_val["target"], token_key, preload=True)
    return None, val_ds


def _load_fold_split(config):
    """Re-derive train/val HF datasets from config (shared logic with main)."""
    fold_split_csv = OmegaConf.select(config, "fold_split_csv", default=None)
    fold_column    = OmegaConf.select(config, "fold_column",    default=None)
    if fold_split_csv and fold_column:
        ds = load_fold_dataset(config.data_dir, fold_split_csv, fold_column)
    else:
        ds = load_local_dataset(config.data_dir,
                                val_split=float(OmegaConf.select(config, "val_split", default=0.15)),
                                seed=int(OmegaConf.select(config, "seed", default=42)))
    return ds["train"], ds["val"]


# ── Main entry point (mirrors curia/trainer.py main()) ───────────────────────


def main(config, trial=None) -> float:
    # Silence stdout on non-rank-0 processes — avoids duplicate print lines with DDP
    if int(os.environ.get("RANK", 0)) != 0:
        sys.stdout = open(os.devnull, "w")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fold_split_csv = OmegaConf.select(config, "fold_split_csv", default=None)
    fold_column    = OmegaConf.select(config, "fold_column",    default=None)
    # Each run gets its own subdirectory so previous runs are never overwritten
    task = OmegaConf.select(config, "task", default="run")
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    run_dir = str(Path(config.output_dir) / f"{task}__{fold_column}__{run_tag}")
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # ── Load local dataset ────────────────────────────────────────────────────


    if fold_split_csv and fold_column:
        print(f"[fold] Using fold split from {fold_split_csv}  column={fold_column}")
        ds = load_fold_dataset(config.data_dir, fold_split_csv, fold_column)
    else:
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
    head_type = OmegaConf.select(config, "model.head_type", default="linear") or "linear"
    use_feature_caching = bool(OmegaConf.select(config, "use_feature_caching", default=False))

    if head_type == "resnet":
        # TokenGridClassifier path: reads full token grid on-the-fly from NPZ
        model, train_dataset, val_dataset = instantiate_resnet_model_and_dataset(
            config, train_dataset, val_dataset
        )
        best_auc = _train_token_grid_model(
            model, train_dataset, val_dataset, config, run_dir, task, timestamp, trial=trial
        )
        if fold_split_csv and int(os.environ.get("RANK", 0)) == 0:
            print("[resnet] Skipping test-set eval (not supported without pooled .pt cache)")
        if sys.stdout != sys.__stdout__:
            sys.stdout.close()
            sys.stdout = sys.__stdout__
        sys.stdout.flush()
        sys.stderr.flush()
        return best_auc

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
    freeze_bb = bool(OmegaConf.select(config, "freeze_backbone", default=True))
    if not freeze_bb and hasattr(model, "backbone"):
        backbone_lr = float(OmegaConf.select(config, "backbone_lr", default=scaled_lr * 0.1))
        param_groups = [
            {"params": model.backbone.parameters(), "lr": backbone_lr},
            {"params": model.linear.parameters(),   "lr": scaled_lr},
        ]
        optimizer = SGD(param_groups, momentum=0.9, weight_decay=weight_decay)
        print(f"Differential LR: backbone={backbone_lr:.2e}  head={scaled_lr:.2e}")
    else:
        optimizer = SGD(model.parameters(), lr=scaled_lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=0)
    print(f"Weight decay: {weight_decay}")

    # ── Fast path: bypass HF Trainer entirely for pre-cached features ────────
    from classification_hf.dataset import _DictDataset as _DD
    if isinstance(train_dataset, _DD):
        # Pure PyTorch loop — all data on GPU, no DataLoader, no NFS writes.
        best_metric = _fast_train_linear(model, train_dataset, val_dataset, config, run_dir, task, timestamp, trial=trial)
        # Test-set inference (only when fold_split_csv is configured)
        if fold_split_csv and int(os.environ.get("RANK", 0)) == 0:
            _cached = getattr(instantiate_cache_model_and_dataset, "_last_pt_cache", None)
            _run_test_eval(model, config, fold_split_csv, run_dir, timestamp, pt_cache=_cached)
        # Free large cache tensor explicitly so GC doesn't delay process exit
        _c = getattr(instantiate_cache_model_and_dataset, "_last_pt_cache", None)
        if _c is not None:
            instantiate_cache_model_and_dataset._last_pt_cache = None
            del _c
        plt.close("all")  # join matplotlib background threads before exit
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        if sys.stdout != sys.__stdout__:
            sys.stdout.close()
            sys.stdout = sys.__stdout__
        sys.stdout.flush()
        sys.stderr.flush()
        return best_metric

    # ── HF Trainer path (full backbone / not pre-cached) ─────────────────────
    training_args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        logging_strategy="steps",
        logging_steps=max(1, steps_per_epoch),
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wbce",
        greater_is_better=False,
        dataloader_num_workers=config.num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=False,
        eval_accumulation_steps=1,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        label_names=["labels"],
        bf16=torch.cuda.is_bf16_supported(),
        gradient_accumulation_steps=int(OmegaConf.select(config, "gradient_accumulation_steps", default=1)),
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
    trainer.remove_callback(PrinterCallback)
    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(_QuietProgressCallback())

    trainer.train()
    trainer.save_model(run_dir)
    save_head(trainer.model, run_dir)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

    if int(os.environ.get("RANK", 0)) == 0:
        _plot_training_history(trainer.state.log_history, run_dir, task)
        _run_final_eval(trainer.model, val_dataset, trainer.args.per_device_eval_batch_size, config, timestamp, run_dir)
        if fold_split_csv and use_feature_caching:
            _cached = getattr(instantiate_cache_model_and_dataset, "_last_pt_cache", None)
            _run_test_eval(trainer.model, config, fold_split_csv, run_dir, timestamp, pt_cache=_cached)
        elif fold_split_csv and not use_feature_caching:
            print("[test] Skipping test-set eval (not supported without feature caching)")

    best_metric = trainer.state.best_metric
    if sys.stdout != sys.__stdout__:
        sys.stdout.close()
        sys.stdout = sys.__stdout__
    sys.stdout.flush()
    sys.stderr.flush()
    return float(best_metric) if best_metric is not None else None
