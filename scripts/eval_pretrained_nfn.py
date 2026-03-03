#!/usr/bin/env python3
"""
Evaluate the pretrained neural_foraminal_narrowing head (raidium/curia)
on pre-extracted 2D slices (PNG) produced by RSNAextractor.py.

Expected directory layout:
    data-dir/
        0/   *.png   (Normal/Mild)
        1/   *.png   (Moderate)
        2/   *.png   (Severe)

Usage:
    python eval_pretrained_nfn.py --data-dir /path/to/patches
"""

import argparse
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

CLASS_NAMES = ["0 Normal/Mild", "1 Moderate", "2 Severe"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_DINO_PATCH = 14
_DILATE_R = _DINO_PATCH // 2 + 1


class Logger:
    """Écrit simultanément dans stdout et dans un fichier."""
    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg)
        self.log_file.write(msg)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def resize_mask(mask_np: np.ndarray, target_size: int) -> torch.Tensor:
    H, W = mask_np.shape
    t = torch.zeros(1, 1, target_size, target_size)

    ys, xs = np.where(mask_np > 0)
    for y, x in zip(ys, xs):
        y_out = min(int(y * target_size / H), target_size - 1)
        x_out = min(int(x * target_size / W), target_size - 1)
        t[0, 0, y_out, x_out] = 1.0

    t = F.max_pool2d(t, kernel_size=2 * _DILATE_R + 1, stride=1, padding=_DILATE_R)
    return t


def run_metrics(labels_np: np.ndarray, probs_np: np.ndarray, preds_np: np.ndarray) -> None:
    n_classes = probs_np.shape[1]

    auc_ovr_macro    = roc_auc_score(labels_np, probs_np, multi_class="ovr", average="macro")
    auc_ovr_weighted = roc_auc_score(labels_np, probs_np, multi_class="ovr", average="weighted")

    print(f"\n{'='*58}")
    print("  AUC — One vs Rest")
    print(f"  {'macro':<20}: {auc_ovr_macro:.4f}")
    print(f"  {'weighted':<20}: {auc_ovr_weighted:.4f}")
    for c in range(n_classes):
        binary = (labels_np == c).astype(int)
        auc_c = roc_auc_score(binary, probs_np[:, c])
        print(f"  class {c} vs rest     : {auc_c:.4f}  (n={int(binary.sum())})")

    print(f"\n  AUC — One vs One (pairwise)")
    auc_ovo_macro = roc_auc_score(labels_np, probs_np, multi_class="ovo", average="macro")
    print(f"  {'macro':<20}: {auc_ovo_macro:.4f}")
    for a, b in combinations(range(n_classes), 2):
        mask = (labels_np == a) | (labels_np == b)
        y_bin = (labels_np[mask] == b).astype(int)
        score = probs_np[mask, b] / (probs_np[mask, a] + probs_np[mask, b] + 1e-9)
        auc_pair = roc_auc_score(y_bin, score)
        print(f"  class {a} vs class {b}    : {auc_pair:.4f}  (n={mask.sum()})")

    print(f"\n  Score (softmax) statistics per class")
    for c in range(n_classes):
        m = labels_np == c
        print(f"  true class {c}  →  mean pred: {probs_np[m].mean(axis=0).round(3)}")

    acc = (preds_np == labels_np).mean()
    print(f"\n  Accuracy : {acc:.4f}")
    print(f"{'='*58}\n")
    print(classification_report(labels_np, preds_np, target_names=CLASS_NAMES, digits=4))

    print("  Confusion matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(labels_np, preds_np)
    header = "         " + "  ".join(f"{n:>8}" for n in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i]:>8}  " + "  ".join(f"{v:>8}" for v in row))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path,
                    default=Path("/home/ge.polymtl.ca/p123239/data/patches_RSNA_raw_with_mask_nfn"),
                    help="Directory with class subfolders 0/ 1/ 2/ containing NPZ slices")
    ap.add_argument("--subfolder", type=str, default="spinal_canal_stenosis",
                    help="Model subfolder in raidium/curia")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--log-dir", type=Path, default=Path("logs"),
                    help="Répertoire parent pour les logs")
    args = ap.parse_args()

    # --- Dossier de log : dataset_name + subfolder ---
    dataset_name = args.data_dir.name
    log_folder = args.log_dir / f"{dataset_name}__{args.subfolder}"
    log_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_folder / f"eval_{timestamp}.log"

    logger = Logger(log_path)
    sys.stdout = logger

    print(f"Log file : {log_path}")
    print(f"Date     : {datetime.now().isoformat()}")
    print(f"Device   : {DEVICE}")
    print(f"Data dir : {args.data_dir}")
    print(f"Subfolder: {args.subfolder}")
    print("Loading model...")

    processor = AutoImageProcessor.from_pretrained("raidium/curia", trust_remote_code=True)
    model = AutoModelForImageClassification.from_pretrained(
        "raidium/curia", subfolder=args.subfolder, trust_remote_code=True
    )
    model.eval().to(DEVICE)

    paths: list[Path] = []
    labels: list[int] = []
    for class_dir in sorted(args.data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        try:
            cls = int(class_dir.name)
        except ValueError:
            continue
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() == ".npz":
                paths.append(f)
                labels.append(cls)

    labels_np = np.array(labels)
    n_classes = len(np.unique(labels_np))
    print(f"\nTotal: {len(paths)} slices  ({n_classes} classes)")
    for c in range(3):
        print(f"  class {c}: {(labels_np == c).sum():>6d}  ({100*(labels_np==c).mean():.1f}%)")

    if len(paths) == 0:
        raise RuntimeError(f"No NPZ files found in {args.data_dir}")

    crop_size = processor.crop_size
    n_missing_mask = 0

    all_probs: list[np.ndarray] = []
    for i in tqdm(range(0, len(paths), args.batch_size), desc="Inference"):
        batch_paths = paths[i:i + args.batch_size]
        batch_data = [np.load(p) for p in batch_paths]
        images_np = [d["slice"].astype(np.float32) for d in batch_data]

        mask_tensors = []
        for d in batch_data:
            if "mask" in d:
                mask_tensors.append(resize_mask(d["mask"], crop_size))
            else:
                n_missing_mask += 1
                mask_tensors.append(None)

        with torch.no_grad():
            inputs = processor(images_np, return_tensors="pt")
            pv = inputs["pixel_values"].to(DEVICE)

            if all(m is not None for m in mask_tensors):
                mask_batch = torch.cat(mask_tensors, dim=0).to(DEVICE)
                try:
                    logits = model(pixel_values=pv, mask=mask_batch)["logits"]
                except NotImplementedError:
                    n_missing_mask += len(batch_paths)
                    logits = model(pixel_values=pv)["logits"]
            else:
                logits = model(pixel_values=pv)["logits"]

            all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())

    if n_missing_mask > 0:
        print(f"[WARN] {n_missing_mask} slices had no mask — ran without mask (CLS token fallback).")

    probs_np = np.concatenate(all_probs, axis=0)
    preds_np = probs_np.argmax(axis=1)

    run_metrics(labels_np, probs_np, preds_np)

    sys.stdout = logger.terminal
    logger.close()
    print(f"\nLog sauvegardé : {log_path}")


if __name__ == "__main__":
    main()