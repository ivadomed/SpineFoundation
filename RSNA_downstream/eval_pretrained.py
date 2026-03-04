#!/usr/bin/env python3
"""
Evaluate a pretrained head from raidium/curia on pre-extracted 2D slices
produced by RSNAextractor.py.

Tasks:
    nfn  — T1w  NeuralForaminalNarrowing   → subfolder: neural_foraminal_narrowing
    ss   — T2w  SubarticularStenosis        → subfolder: subarticular_stenosis
    scs  — T2w  SpinalCanalStenosis         → subfolder: spinal_canal_stenosis

Expected directory layout:
    data-dir/
        0/   *.npz   (Normal/Mild)
        1/   *.npz   (Moderate)
        2/   *.npz   (Severe)

Usage:
    python eval_pretrained.py --task nfn --data-dir /path/to/patches
"""

import argparse
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

CLASS_NAMES = ["0 Normal/Mild", "1 Moderate", "2 Severe"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_DATA_ROOT = Path("/home/ge.polymtl.ca/p123239/data")
TASK_CONFIG = {
    "nfn": {
        "subfolder": "neural_foraminal_narrowing",
        "data_dir": _DATA_ROOT / "patches_RSNA_raw_with_mask_nfn",
    },
    "ss": {
        "subfolder": "subarticular_stenosis",
        "data_dir": _DATA_ROOT / "patches_RSNA_raw_with_mask_ss",
    },
    "scs": {
        "subfolder": "spinal_canal_stenosis",
        "data_dir": _DATA_ROOT / "patches_RSNA_raw_with_mask_scs",
    },
}


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


def make_mask_transform(mask_np: np.ndarray, target_size: int, dilation_radius: int = 0) -> torch.Tensor:
    """Reproduit make_mask_transform() de raidium/curia/trainer.py :
    NumpyToTensor → AdaptativeResizeMask (bilinear antialias, seuil adaptatif).
    dilation_radius : dilation en pixels dans l'espace target_size (via max_pool2d).
    Retourne (1, 1, target_size, target_size) float32.
    """
    t = torch.tensor(mask_np).unsqueeze(0).float()  # (1, H, W)
    t = t.unsqueeze(0)  # (1, 1, H, W)
    t = TF.resize(t, [target_size, target_size],
                  interpolation=TF.InterpolationMode.BILINEAR,
                  antialias=True)

    if dilation_radius > 0:
        k = 2 * dilation_radius + 1
        t = F.max_pool2d(t, kernel_size=k, stride=1, padding=dilation_radius)

    t = t.squeeze(0)  # (1, target_size, target_size)
    mask = t > 0.5
    if mask.sum() == 0:
        new_threshold = t.max() * 0.5
        mask = t > new_threshold

    return mask.float().unsqueeze(0)  # (1, 1, target_size, target_size)


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
    ap.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["nfn", "ss", "scs"],
        help="nfn: NeuralForaminalNarrowing | ss: SubarticularStenosis | scs: SpinalCanalStenosis",
    )
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="Directory with class subfolders 0/ 1/ 2/ (default: task-specific)")
    ap.add_argument("--subfolder", type=str, default=None,
                    help="Model subfolder in raidium/curia (default: task-specific)")
    ap.add_argument("--model-name", type=str, default="raidium/curia",
                    help="Path or HF repo of the curia model (default: raidium/curia)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--dilation-radius", type=int, default=0,
                    help="Dilate the mask by this many pixels in 512×512 space (via max_pool2d). "
                         "Each DINOv2 patch = 16px, so radius=16 adds ~1 patch of context. 0 = no dilation.")
    ap.add_argument("--log-dir", type=Path, default=Path("logs"),
                    help="Répertoire parent pour les logs")
    args = ap.parse_args()

    cfg = TASK_CONFIG[args.task]
    if args.data_dir is None:
        args.data_dir = cfg["data_dir"]
    if args.subfolder is None:
        args.subfolder = cfg["subfolder"]

    # --- Dossier de log : dataset_name + subfolder (+ dilation) ---
    dataset_name = args.data_dir.name
    dil_suffix = f"__dil{args.dilation_radius}" if args.dilation_radius > 0 else ""
    log_folder = args.log_dir / f"{dataset_name}__{args.subfolder}{dil_suffix}"
    log_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_folder / f"eval_{timestamp}.log"

    logger = Logger(log_path)
    sys.stdout = logger

    print(f"Log file : {log_path}")
    print(f"Date     : {datetime.now().isoformat()}")
    print(f"Device   : {DEVICE}")
    print(f"Data dir : {args.data_dir}")
    print(f"Model    : {args.model_name}")
    print(f"Subfolder: {args.subfolder}")
    print(f"Dilation : {args.dilation_radius}px" if args.dilation_radius > 0 else "Dilation : none")
    print("Loading model...")

    processor = AutoImageProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForImageClassification.from_pretrained(
        args.model_name, subfolder=args.subfolder, trust_remote_code=True
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
                mask_tensors.append(make_mask_transform(d["mask"], crop_size, args.dilation_radius))
            else:
                n_missing_mask += 1
                mask_tensors.append(None)
                print(f"[WARN] File {d} has no mask — will run without mask (CLS token fallback).")

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
                    print(f"[WARN] Model does not support masks — ran without mask for this batch (CLS token fallback).")
            else:
                logits = model(pixel_values=pv)["logits"]
                print(f"[WARN] Some slices in this batch had no mask — ran without mask (CLS token fallback).")

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