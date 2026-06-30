"""
Evaluate test-set predictions saved by the trainer.

Reads one or more test_predictions.npz files (logits + labels), computes:
  - Cross-entropy loss
  - Macro AUC (OvR)
  - Per-class and overall confusion matrix (counts + normalised)

Usage:
    # Single run
    python -m classification_hf.eval_test \
        --pred /path/to/run_dir/test_predictions.npz \
        --task nfn

    # Pool multiple runs (stacks logits before computing metrics)
    python -m classification_hf.eval_test \
        --pred /path/to/outputs_cls/rsna_nfn_fold \
        --task nfn

    # Evaluate all three tasks at once
    python -m classification_hf.eval_test --task nfn scs ss \
        --pred /path/to/outputs_cls/rsna_nfn_fold \
               /path/to/outputs_cls/rsna_scs_fold \
               /path/to/outputs_cls/rsna_ss_fold
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

CLASS_NAMES = ["Normal/Mild", "Moderate", "Severe"]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _find_pred_files(path: Path) -> list[Path]:
    """Accept either a direct .npz file or a directory tree."""
    if path.is_file() and path.suffix == ".npz":
        return [path]
    found = sorted(path.glob("**/test_predictions.npz"))
    if not found:
        raise FileNotFoundError(f"No test_predictions.npz found under {path}")
    return found


def _load_predictions(pred_files: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Stack logits and labels from one or more prediction files."""
    all_logits, all_labels = [], []
    for pf in pred_files:
        d = np.load(str(pf))
        all_logits.append(d["logits"].astype(np.float32))
        all_labels.append(d["labels"].astype(np.int64))
        print(f"  loaded {pf}  ({len(d['labels'])} samples)")
    return np.concatenate(all_logits), np.concatenate(all_labels)


def cross_entropy_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    log_probs = torch.log_softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)
    loss = torch.nn.functional.nll_loss(log_probs, torch.tensor(labels, dtype=torch.long))
    return float(loss)


# ── Core evaluation ───────────────────────────────────────────────────────────


def evaluate(logits: np.ndarray, labels: np.ndarray, task: str, out_dir: Path) -> dict:
    proba = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()
    preds = np.argmax(logits, axis=-1)

    ce  = cross_entropy_loss(logits, labels)
    acc = accuracy_score(labels, preds)
    bal = balanced_accuracy_score(labels, preds)
    f1m = f1_score(labels, preds, average="macro",    zero_division=0)
    f1w = f1_score(labels, preds, average="weighted", zero_division=0)
    qwk = cohen_kappa_score(labels, preds, weights="quadratic")
    kap = cohen_kappa_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)

    try:
        auc_macro    = roc_auc_score(labels, proba, multi_class="ovr", average="macro")
        auc_weighted = roc_auc_score(labels, proba, multi_class="ovr", average="weighted")
        auc_per_class = [
            roc_auc_score((labels == c).astype(int), proba[:, c])
            for c in range(proba.shape[1])
        ]
    except ValueError:
        auc_macro = auc_weighted = float("nan")
        auc_per_class = [float("nan")] * proba.shape[1]

    report = classification_report(
        labels, preds,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    # ── Console output ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Task : {task}   |   n={len(labels)}")
    print(f"  Cross-entropy     : {ce:.4f}")
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Balanced accuracy : {bal:.4f}")
    print(f"  F1 macro          : {f1m:.4f}")
    print(f"  F1 weighted       : {f1w:.4f}")
    print(f"  QWK               : {qwk:.4f}")
    print(f"  Kappa             : {kap:.4f}")
    print(f"  MCC               : {mcc:.4f}")
    print(f"  AUC OvR macro     : {auc_macro:.4f}")
    print(f"  AUC OvR weighted  : {auc_weighted:.4f}")
    for c, a in enumerate(auc_per_class):
        print(f"    AUC {CLASS_NAMES[c]:<12}: {a:.4f}")
    print(f"\n{report}")

    # ── Confusion matrices ────────────────────────────────────────────────────
    cm_counts = confusion_matrix(labels, preds)
    cm_norm   = confusion_matrix(labels, preds, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Test-set confusion matrix — {task}", fontsize=13)

    for ax, cm, title, fmt in [
        (axes[0], cm_counts, "Counts",       "d"),
        (axes[1], cm_norm,   "Row-normalised", ".2f"),
    ]:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
        disp.plot(ax=ax, colorbar=False, values_format=fmt)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    cm_path = out_dir / f"confusion_matrix_{task}.png"
    plt.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved → {cm_path}")

    # ── JSON summary ──────────────────────────────────────────────────────────
    result = {
        "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "task":               task,
        "n_samples":          int(len(labels)),
        "cross_entropy_loss": round(ce,           4),
        "accuracy":           round(acc,          4),
        "balanced_accuracy":  round(bal,          4),
        "f1_macro":           round(f1m,          4),
        "f1_weighted":        round(f1w,          4),
        "qwk":                round(qwk,          4),
        "kappa":              round(kap,          4),
        "mcc":                round(mcc,          4),
        "auc_ovr_macro":      round(auc_macro,    4),
        "auc_ovr_weighted":   round(auc_weighted, 4),
        "auc_per_class":      {CLASS_NAMES[c]: round(a, 4) for c, a in enumerate(auc_per_class)},
        "confusion_matrix_counts":     cm_counts.tolist(),
        "confusion_matrix_normalised": [[round(v, 4) for v in row] for row in cm_norm.tolist()],
        "classification_report":       report,
    }
    json_path = out_dir / f"test_eval_{task}.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"JSON summary saved   → {json_path}")

    return result


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Evaluate test-set logits")
    parser.add_argument(
        "--pred", nargs="+", required=True, type=Path,
        help="Path(s) to test_predictions.npz or to a directory tree containing them. "
             "Must match --task order if multiple tasks are given.",
    )
    parser.add_argument(
        "--task", nargs="+", required=True,
        help="Task name(s) (e.g. nfn scs ss). Must match --pred order.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Directory to write plots and JSON (default: same folder as first pred file).",
    )
    args = parser.parse_args()

    if len(args.pred) != len(args.task):
        parser.error("--pred and --task must have the same number of arguments")

    for pred_path, task in zip(args.pred, args.task):
        print(f"\n── Task: {task}  pred: {pred_path} ─────────────────────────────")
        pred_files = _find_pred_files(pred_path)
        print(f"   {len(pred_files)} prediction file(s):")
        logits, labels = _load_predictions(pred_files)
        print(f"   Total: {len(labels)} samples  "
              f"class distribution: {np.bincount(labels).tolist()}")

        out_dir = args.out_dir or (pred_files[0].parent if pred_files[0].is_file()
                                   else pred_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        evaluate(logits, labels, task, out_dir)


if __name__ == "__main__":
    main()
