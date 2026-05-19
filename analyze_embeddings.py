#!/usr/bin/env python3
"""
Spine Foundation Model – Embedding Analysis
Extracts ViT CLS-token embeddings, then runs UMAP + linear probing
stratified by contrast, orientation, body part, and pathology.

Usage:
    python analyze_embeddings.py \
        --data_dir /path/to/dataset_root \
        --model raidium/curia \
        --output_dir ./analysis_output \
        --split both \
        --max_per_dataset 200
"""

import argparse
import os
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from PIL import Image
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ─── Dataset-level metadata ──────────────────────────────────────────────────

DATASET_BODY_PART: dict[str, str] = {
    "als-basel-ramira":               "cervical",
    "bavaria-quebec-spine-ms":        "cervical",
    "beijing-tumor":                  "mixed",
    "canproco":                       "cervical",
    "data-multi-subject":             "cervical",
    "data-single-subject":            "cervical",
    "dcm-brno":                       "cervical",
    "dcm-oklahoma":                   "cervical",
    "dcm-zurich":                     "cervical",
    "dcm-zurich-lesions":             "cervical",
    "dcm-zurich-lesions-20231115":    "cervical",
    "gmseg-challenge-2016":           "cervical",
    "hc-leipzig-7t-mp2rage":          "cervical",
    "hc-lumbar-shanghai":             "lumbar",
    "hc-lumbar-zurich":               "lumbar",
    "hc-ucsf-psir":                   "cervical",
    "inspired":                       "cervical",
    "lbp-lumbar-usf-2024":            "lumbar",
    "lbp-lumbar-usf-2025":            "lumbar",
    "lumbar-epfl":                    "lumbar",
    "lumbar-marseille":               "lumbar",
    "lumbar-nusantara":               "lumbar",
    "lumbar-rsna-challenge-2024":     "lumbar",
    "lumbar-vanderbilt":              "lumbar",
    "marseille-rootlets":             "rootlets",
    "ms-barcelona-psir":              "cervical",
    "ms-basel-2018":                  "cervical",
    "ms-basel-2020":                  "cervical",
    "ms-dresden-mp2rage-2025":        "cervical",
    "ms-karolinska-2020":             "cervical",
    "ms-mayo-critical-lesions-2025":  "cervical",
    "ms-mayo-critical-lesions-2026":  "cervical",
    "ms-multi-spine-challenge-2024":  "mixed",
    "ms-nyu":                         "cervical",
    "msseg_challenge_2016":           "cervical",
    "msseg_challenge_2021":           "cervical",
    "ms-ucsf-2025":                   "cervical",
    "nih-ms-mp2rage":                 "cervical",
    "nisci-trial":                    "cervical",
    "nrc-lumbar-balgrist":            "lumbar",
    "pd-mcgill":                      "cervical",
    "sci-colorado":                   "cervical",
    "sci-paris":                      "cervical",
    "sci-zurich":                     "cervical",
    "sct-testing-large":              "mixed",
    "spider-challenge-2023":          "lumbar",
    "twh-rootlets":                   "rootlets",
}

DATASET_PATHOLOGY: dict[str, str] = {
    "als-basel-ramira":               "ALS",
    "bavaria-quebec-spine-ms":        "MS",
    "beijing-tumor":                  "tumor",
    "canproco":                       "MS",
    "data-multi-subject":             "healthy",
    "data-single-subject":            "healthy",
    "dcm-brno":                       "DCM",
    "dcm-oklahoma":                   "DCM",
    "dcm-zurich":                     "DCM",
    "dcm-zurich-lesions":             "DCM",
    "dcm-zurich-lesions-20231115":    "DCM",
    "gmseg-challenge-2016":           "healthy",
    "hc-leipzig-7t-mp2rage":          "healthy",
    "hc-lumbar-shanghai":             "healthy",
    "hc-lumbar-zurich":               "healthy",
    "hc-ucsf-psir":                   "healthy",
    "inspired":                       "SCI",
    "lbp-lumbar-usf-2024":            "LBP",
    "lbp-lumbar-usf-2025":            "LBP",
    "lumbar-epfl":                    "healthy",
    "lumbar-marseille":               "healthy",
    "lumbar-nusantara":               "healthy",
    "lumbar-rsna-challenge-2024":     "mixed",
    "lumbar-vanderbilt":              "healthy",
    "marseille-rootlets":             "healthy",
    "ms-barcelona-psir":              "MS",
    "ms-basel-2018":                  "MS",
    "ms-basel-2020":                  "MS",
    "ms-dresden-mp2rage-2025":        "MS",
    "ms-karolinska-2020":             "MS",
    "ms-mayo-critical-lesions-2025":  "MS",
    "ms-mayo-critical-lesions-2026":  "MS",
    "ms-multi-spine-challenge-2024":  "MS",
    "ms-nyu":                         "MS",
    "msseg_challenge_2016":           "MS",
    "msseg_challenge_2021":           "MS",
    "ms-ucsf-2025":                   "MS",
    "nih-ms-mp2rage":                 "MS",
    "nisci-trial":                    "SCI",
    "nrc-lumbar-balgrist":            "healthy",
    "pd-mcgill":                      "PD",
    "sci-colorado":                   "SCI",
    "sci-paris":                      "SCI",
    "sci-zurich":                     "SCI",
    "sct-testing-large":              "mixed",
    "spider-challenge-2023":          "mixed",
    "twh-rootlets":                   "healthy",
}


# ─── File-level metadata parsing ─────────────────────────────────────────────

_CONTRAST_RE = re.compile(r"^sub-[^_]+_(.*)")
_SPACING_RE  = re.compile(r"sp(\d+)x(\d+)")

SPACING_BINS = [
    ("lt0.35",  0.0,  0.35),
    ("0.35-0.5", 0.35, 0.5),
    ("0.5-0.7",  0.5,  0.7),
    ("0.7-0.8",  0.7,  0.8),
    ("0.8-1.0",  0.8,  1.0),
    ("1.0",      1.0,  1.05),
    ("gt1.05",   1.05, float("inf")),
]


def parse_filename(filename: str) -> dict:
    """Parse metadata encoded in the PNG filename.

    Convention: {subject}__{subject}_{contrast}__{orientation}__{slice}__{t}__{sp}.png
    """
    stem = Path(filename).stem
    parts = stem.split("__")

    contrast = "unknown"
    orientation = "unknown"
    spacing_mm = float("nan")

    if len(parts) >= 2:
        m = _CONTRAST_RE.match(parts[1])
        if m:
            contrast = m.group(1)

    if len(parts) >= 3:
        orientation = parts[2]

    m = _SPACING_RE.search(stem)
    if m:
        sx = int(m.group(1)) / 1000.0
        sy = int(m.group(2)) / 1000.0
        spacing_mm = (sx + sy) / 2.0

    subject = parts[0] if parts else "unknown"

    return {"subject": subject, "contrast": contrast,
            "orientation": orientation, "spacing_mm": spacing_mm}


# ─── Data collection ──────────────────────────────────────────────────────────

def collect_files(data_dir: Path, splits: list[str],
                  max_per_dataset: int | None) -> pd.DataFrame:
    records = []
    for split in splits:
        split_dir = data_dir / "image" / split
        if not split_dir.exists():
            print(f"  [warn] {split_dir} not found, skipping")
            continue
        for dataset_dir in sorted(split_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name
            pngs = sorted(dataset_dir.glob("*.png"))
            if max_per_dataset is not None:
                rng = np.random.default_rng(42)
                pngs = list(rng.choice(pngs, min(len(pngs), max_per_dataset),
                                       replace=False))
            for p in pngs:
                meta = parse_filename(p.name)
                records.append({
                    "path": str(p),
                    "split": split,
                    "dataset": dataset,
                    "body_part": DATASET_BODY_PART.get(dataset, "unknown"),
                    "pathology": DATASET_PATHOLOGY.get(dataset, "unknown"),
                    **meta,
                })

    df = pd.DataFrame(records)
    return df


# ─── Model loading ────────────────────────────────────────────────────────────

def model_slug(model_name: str) -> str:
    return os.path.normpath(model_name).replace("/", "_").lstrip("_.")


def load_model(model_name: str, device: torch.device):
    print(f"Loading model  : {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
    model = torch.compile(model)
    return model, processor


# ─── Embedding extraction ─────────────────────────────────────────────────────

class _ImageDataset(Dataset):
    def __init__(self, paths: list[str], img_mode: str):
        self.paths = paths
        self.img_mode = img_mode

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return np.array(Image.open(self.paths[idx]).convert(self.img_mode))


@torch.no_grad()
def extract_embeddings(df: pd.DataFrame, model, processor,
                       device: torch.device, batch_size: int,
                       num_workers: int = 8) -> np.ndarray:
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    img_mode = "L" if getattr(inner.config, "num_channels", 3) == 1 else "RGB"

    def collate(images):
        inputs = processor(images=images, return_tensors="pt")
        # drop any spurious size-1 dims (e.g. (B,3,1,H,W) → (B,3,H,W))
        if "pixel_values" in inputs and inputs["pixel_values"].dim() > 4:
            pv = inputs["pixel_values"]
            new_shape = [pv.shape[0]] + [s for s in pv.shape[1:] if s != 1]
            inputs["pixel_values"] = pv.reshape(new_shape)
        return inputs

    loader = DataLoader(
        _ImageDataset(df["path"].tolist(), img_mode),
        batch_size=batch_size, num_workers=num_workers,
        collate_fn=collate, pin_memory=True, prefetch_factor=2,
    )

    use_fp16 = device.type == "cuda"
    all_embs: list[np.ndarray] = []
    for inputs in tqdm(loader, desc="Embeddings"):
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
            out = model(**inputs)
        cls = out.last_hidden_state[:, 0, :].cpu().float().numpy()
        all_embs.append(cls)

    return np.vstack(all_embs)


# ─── UMAP ────────────────────────────────────────────────────────────────────

def run_umap(embeddings: np.ndarray,
             n_neighbors: int, min_dist: float) -> np.ndarray:
    print("Running UMAP…")
    try:
        import ctypes, os as _os
        _conda = "/home/ge.polymtl.ca/p123239/.conda/envs/FM"
        _os.environ.setdefault("CONDA_PREFIX", _conda)
        ctypes.CDLL(f"{_conda}/lib/libnvJitLink.so.13", mode=ctypes.RTLD_GLOBAL)
        from cuml.manifold import UMAP as cuUMAP
        import cupy as cp
        print("  Using GPU UMAP (cuML)")
        X_gpu = cp.array(embeddings.astype(np.float32))
        reducer = cuUMAP(
            n_neighbors=n_neighbors, min_dist=min_dist,
            n_components=2, output_type="numpy",
        )
        return reducer.fit_transform(X_gpu)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  cuML unavailable ({type(e).__name__}: {e}), falling back to CPU UMAP")
        reducer = umap.UMAP(
            n_neighbors=n_neighbors, min_dist=min_dist,
            n_components=2, random_state=42, verbose=True,
            n_jobs=-1, low_memory=False,
        )
        return reducer.fit_transform(embeddings)


def _make_colormap(labels: list[str]) -> dict[str, tuple]:
    unique = sorted(set(labels))
    n = len(unique)
    cmap = plt.get_cmap("tab20" if n <= 20 else "hsv")
    return {lbl: cmap(i / max(n - 1, 1)) for i, lbl in enumerate(unique)}


def plot_umap(coords: np.ndarray, df: pd.DataFrame,
              output_dir: Path, columns: list[str]) -> None:
    for col in columns:
        if col not in df.columns:
            continue
        labels = df[col].fillna("unknown").astype(str).tolist()
        color_map = _make_colormap(labels)

        fig, ax = plt.subplots(figsize=(13, 9))
        for lbl in sorted(set(labels)):
            mask = np.array([l == lbl for l in labels])
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=[color_map[lbl]], label=lbl,
                       s=6, alpha=0.65, linewidths=0)

        ax.set_title(f"UMAP — colored by {col}", fontsize=14)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        n_unique = len(set(labels))
        if n_unique <= 35:
            ax.legend(markerscale=3, fontsize=7,
                      bbox_to_anchor=(1.01, 1), loc="upper left")
        plt.tight_layout()
        out_path = output_dir / f"umap_{col}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path.name}")


# ─── Linear probing ───────────────────────────────────────────────────────────

def _fit_linear_probe(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    n_classes: int, device: torch.device,
    epochs: int = 300, lr: float = 1e-2,
    batch_size: int = 4096, weight_decay: float = 1e-4,
) -> np.ndarray:
    n_features = X_train.shape[1]

    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1, counts)
    weights = torch.tensor(1.0 / counts, dtype=torch.float32, device=device)

    model = nn.Linear(n_features, n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=weights)

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.long, device=device)

    model.train()
    n = len(X_t)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            criterion(model(X_t[idx]), y_t[idx]).backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_v = torch.tensor(X_val, dtype=torch.float32, device=device)
        pred = model(X_v).argmax(dim=1).cpu().numpy()
    return pred


def linear_probing(embeddings: np.ndarray, df: pd.DataFrame,
                   columns: list[str], output_dir: Path,
                   holdout_emb: np.ndarray | None = None,
                   holdout_df: pd.DataFrame | None = None,
                   n_splits: int = 5, suffix: str = "") -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Linear probing device: {device}")

    use_group_cv = "subject" in df.columns and df["subject"].nunique() >= n_splits
    if use_group_cv:
        groups = (df["dataset"] + "__" + df["subject"]).values
        n_groups = len(set(groups))
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        print(f"  Using StratifiedGroupKFold ({n_groups} unique dataset__subject groups, {n_splits} folds)")
    else:
        groups = None
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        print(f"  Using StratifiedKFold (no subject info)")

    rows = []

    for col in columns:
        if col not in df.columns:
            continue
        y_raw = df[col].fillna("unknown").astype(str)
        classes = sorted(y_raw.unique())
        if len(classes) < 2:
            print(f"  {col}: only 1 class – skipping")
            continue

        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        n_classes = len(classes)

        split_iter = cv.split(embeddings, y, groups=groups) if use_group_cv else cv.split(embeddings, y)
        scores = []
        all_y_true, all_y_pred = [], []
        for train_idx, val_idx in split_iter:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(embeddings[train_idx])
            X_val   = scaler.transform(embeddings[val_idx])
            pred = _fit_linear_probe(
                X_train, y[train_idx],
                X_val,   y[val_idx],
                n_classes, device,
            )
            scores.append(balanced_accuracy_score(y[val_idx], pred))
            all_y_true.append(y[val_idx])
            all_y_pred.append(pred)

        mean, std = float(np.mean(scores)), float(np.std(scores))
        chance = 1.0 / len(classes)
        print(f"  {col:30s}  CV bal_acc={mean:.3f} ± {std:.3f}  "
              f"(chance={chance:.3f}, n_classes={len(classes)})", end="")

        # ── Holdout evaluation ──────────────────────────────────────────────
        holdout_bal_acc = None
        if holdout_emb is not None and holdout_df is not None and col in holdout_df.columns:
            y_holdout_raw = holdout_df[col].fillna("unknown").astype(str)
            known_mask = y_holdout_raw.isin(le.classes_).values

            scaler_final = StandardScaler()
            X_train_final = scaler_final.fit_transform(embeddings)

            if known_mask.sum() >= 1:
                y_holdout = le.transform(y_holdout_raw[known_mask])
                X_holdout = scaler_final.transform(holdout_emb[known_mask])
                pred_holdout = _fit_linear_probe(
                    X_train_final, y,
                    X_holdout, y_holdout,
                    n_classes, device,
                )
                holdout_bal_acc = float(balanced_accuracy_score(y_holdout, pred_holdout))
                print(f"  holdout_bal_acc={holdout_bal_acc:.3f}", end="")

                cm_holdout = confusion_matrix(y_holdout, pred_holdout, labels=np.arange(n_classes))
                if n_classes <= 20:
                    _plot_confusion_matrix(cm_holdout, le.classes_, output_dir, col, suffix + "_holdout")

            # For unseen holdout labels (e.g. holdout datasets), show where the model places them
            unseen_mask = ~known_mask
            if unseen_mask.sum() >= 1:
                X_unseen = scaler_final.transform(holdout_emb[unseen_mask])
                pred_unseen = _fit_linear_probe(
                    X_train_final, y,
                    X_unseen, np.zeros(unseen_mask.sum(), dtype=int),
                    n_classes, device,
                )
                pred_labels = le.inverse_transform(pred_unseen)
                unseen_true = y_holdout_raw[unseen_mask].values
                dist_df = (
                    pd.DataFrame({"true": unseen_true, "predicted_as": pred_labels})
                    .groupby(["true", "predicted_as"])
                    .size()
                    .reset_index(name="count")
                )
                dist_df["frac"] = dist_df.groupby("true")["count"].transform(lambda x: x / x.sum())
                dist_df["frac"] = dist_df["frac"].round(3)
                out_path = output_dir / f"linear_probing_{col}_unseen_holdout{suffix}.csv"
                dist_df.to_csv(out_path, index=False)
                print(f"\n    Unseen holdout prediction distribution saved to {out_path.name}")
                for true_ds, grp in dist_df.groupby("true"):
                    top = grp.sort_values("frac", ascending=False).head(3)
                    top_str = ", ".join(f"{r.predicted_as}({r.frac:.0%})" for _, r in top.iterrows())
                    print(f"    {true_ds} → {top_str}")
        print()

        y_true_all = np.concatenate(all_y_true)
        y_pred_all = np.concatenate(all_y_pred)

        cm = confusion_matrix(y_true_all, y_pred_all, labels=np.arange(n_classes))

        report = classification_report(
            y_true_all, y_pred_all,
            labels=np.arange(n_classes),
            target_names=le.classes_,
            output_dict=True, zero_division=0,
        )

        # per-class specificity = TN / (TN + FP)
        specificity_per_class = {}
        for i, cls in enumerate(le.classes_):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            specificity_per_class[cls] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        per_class_records = []
        for cls in le.classes_:
            r = report.get(cls, {})
            per_class_records.append({
                "class":       cls,
                "precision":   round(r.get("precision", 0), 4),
                "recall":      round(r.get("recall", 0), 4),
                "specificity": round(specificity_per_class[cls], 4),
                "f1_score":    round(r.get("f1-score", 0), 4),
                "support":     int(r.get("support", 0)),
            })
        per_class_df = pd.DataFrame(per_class_records)
        per_class_df.to_csv(output_dir / f"linear_probing_{col}_per_class{suffix}.csv", index=False)

        if n_classes <= 20:
            _plot_confusion_matrix(cm, le.classes_, output_dir, col, suffix)

        acc  = report.get("accuracy", 0.0)
        row = {
            "attribute":            col,
            "n_classes":            len(classes),
            "classes":              ", ".join(classes),
            "chance":               round(chance, 4),
            "accuracy":             round(acc, 4),
            "bal_acc_mean":         round(mean, 4),
            "bal_acc_std":          round(std, 4),
            "delta_over_chance":    round(mean - chance, 4),
            "macro_precision":      round(report["macro avg"]["precision"], 4),
            "macro_recall":         round(report["macro avg"]["recall"], 4),
            "macro_f1":             round(report["macro avg"]["f1-score"], 4),
            "weighted_precision":   round(report["weighted avg"]["precision"], 4),
            "weighted_recall":      round(report["weighted avg"]["recall"], 4),
            "weighted_f1":          round(report["weighted avg"]["f1-score"], 4),
            "macro_specificity":    round(np.mean(list(specificity_per_class.values())), 4),
        }
        if holdout_bal_acc is not None:
            row["holdout_bal_acc"] = round(holdout_bal_acc, 4)
        rows.append(row)

    return pd.DataFrame(rows)


def _plot_confusion_matrix(cm: np.ndarray, class_names: np.ndarray,
                            output_dir: Path, col: str, suffix: str = "") -> None:
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.7), max(5, n * 0.6)))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    thresh = 0.5
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center", fontsize=6,
                    color="white" if cm_norm[i, j] > thresh else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    title = f"Confusion matrix — {col} (row-normalized)"
    if suffix:
        title += f"  [{suffix.lstrip('_')}]"
    ax.set_title(title)
    plt.tight_layout()
    out_path = output_dir / f"confusion_matrix_{col}{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_linear_probing(results: pd.DataFrame, output_dir: Path,
                        suffix: str = "", title_suffix: str = "") -> None:
    if results.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, len(results) * 1.4), 5))
    x = np.arange(len(results))
    bars = ax.bar(x, results["bal_acc_mean"],
                  yerr=results["bal_acc_std"],
                  capsize=5, color="steelblue", alpha=0.85)

    for i, row in results.iterrows():
        ax.hlines(row["chance"], i - 0.4, i + 0.4,
                  colors="tomato", linestyles="--", linewidth=1.5)

    # phantom lines for legend
    ax.hlines([], [], [], colors="tomato", linestyles="--",
              linewidth=1.5, label="chance level (per attribute)")

    for bar, val in zip(bars, results["bal_acc_mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + results["bal_acc_std"].max() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(results["attribute"], rotation=25, ha="right")
    ax.set_ylabel("Balanced accuracy (5-fold CV)")
    ax.set_title(f"Linear probing separability by attribute{title_suffix}")
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.tight_layout()
    out_path = output_dir / f"linear_probing{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spine Foundation – UMAP + linear probing analysis"
    )
    parser.add_argument("--data_dir",    required=True,
                        help="Root dir containing image/ and label/ sub-dirs")
    parser.add_argument("--model",       required=True,
                        help="HuggingFace model id, e.g. raidium/curia")
    parser.add_argument("--output_dir",  default="./analysis_output")
    parser.add_argument("--split",       default="train",
                        choices=["train", "val", "both"])
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--max_per_dataset", type=int, default=None,
                        help="Cap images per dataset (useful for quick tests)")
    parser.add_argument("--orientation", default=None,
                        help="Keep only this orientation (e.g. ax, sag, cor)")
    parser.add_argument("--datasets", default=None,
                        help="Comma-separated list of dataset names to keep")
    parser.add_argument("--holdout_datasets", default=None,
                        help="Comma-separated dataset names used as test-only holdout (never in train)")
    parser.add_argument("--umap_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--num_workers",  type=int, default=8,
                        help="DataLoader worker processes for image loading")
    parser.add_argument("--cache_dir",   default=None,
                        help="Dir for embedding cache (.npy/.csv). Defaults to --output_dir if omitted.")
    parser.add_argument("--no_umap",            action="store_true")
    parser.add_argument("--no_linear_probing",  action="store_true")
    parser.add_argument("--recompute",          action="store_true",
                        help="Recompute embeddings even if cached files exist")
    parser.add_argument("--device",      default=None,
                        help="'cuda', 'cpu', or 'mps'. Auto-detected if omitted.")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val"] if args.split == "both" else [args.split]

    # ── Collect files ──
    print("Scanning files…")
    df = collect_files(Path(args.data_dir), splits, args.max_per_dataset)
    if df.empty:
        print("No PNG files found. Check --data_dir.")
        return
    if args.orientation:
        df = df[df["orientation"] == args.orientation].reset_index(drop=True)
        print(f"Filtered to orientation='{args.orientation}': {len(df):,} images")
        if df.empty:
            print("No images left after filtering. Check --orientation value.")
            return
    if args.datasets:
        keep = [d.strip() for d in args.datasets.split(",")]
        df = df[df["dataset"].isin(keep)].reset_index(drop=True)
        print(f"Filtered to {len(keep)} datasets: {len(df):,} images")
        if df.empty:
            print("No images left after filtering. Check --datasets value.")
            return
    print(f"Found {len(df):,} images across {df['dataset'].nunique()} datasets")
    datasets_used = sorted(df["dataset"].unique())
    print(df[["split", "dataset", "contrast", "orientation",
              "body_part", "pathology"]].describe(include="all").to_string())

    # ── Model + embeddings ──
    slug        = model_slug(args.model)
    orient_slug = f"_{args.orientation}" if args.orientation else ""
    cache_emb   = cache_dir / f"embeddings_{slug}{orient_slug}.npy"
    cache_meta  = cache_dir / f"metadata_{slug}{orient_slug}.csv"
    if not args.recompute and cache_emb.exists() and cache_meta.exists():
        print(f"Loading cached embeddings from {cache_emb} …")
        embeddings = np.load(cache_emb)
        df_cache = pd.read_csv(cache_meta)
        # re-apply filters so df and embeddings stay aligned
        mask = pd.Series([True] * len(df_cache))
        if args.orientation:
            mask &= df_cache["orientation"] == args.orientation
        if args.datasets:
            keep = [d.strip() for d in args.datasets.split(",")]
            mask &= df_cache["dataset"].isin(keep)
        df = df_cache[mask].reset_index(drop=True)
        embeddings = embeddings[mask.values]
        print(f"Embedding shape after filtering: {embeddings.shape}")
        parsed = df["path"].apply(lambda p: parse_filename(Path(p).name))
        if "spacing_mm" not in df.columns:
            df["spacing_mm"] = parsed.apply(lambda d: d.get("spacing_mm", float("nan")))
        if "subject" not in df.columns:
            df["subject"] = parsed.apply(lambda d: d.get("subject", "unknown"))
    else:
        model, processor = load_model(args.model, device)
        embeddings = extract_embeddings(df, model, processor, device, args.batch_size, args.num_workers)
        print(f"Embedding shape: {embeddings.shape}")
        np.save(cache_emb, embeddings)
        df.to_csv(cache_meta, index=False)
        print("Saved embeddings.npy + metadata.csv")

    if "spacing_mm" in df.columns:
        def _bin_spacing(v):
            for name, lo, hi in SPACING_BINS:
                if lo <= v < hi:
                    return name
            return "unknown"
        df["spacing_bin"] = df["spacing_mm"].apply(_bin_spacing)

    # ── Contrast type grouping ──
    def _map_contrast_type(c: str) -> str | None:
        if "T2star" in c: return "T2star"
        if "T2w"    in c: return "T2w"
        if "T1w"    in c: return "T1w"
        if "UNIT1"  in c: return "UNIT1"
        if "MP2RAGE" in c: return "MP2RAGE"
        return None
    df["contrast_type"] = df["contrast"].apply(_map_contrast_type)
    mask = df["contrast_type"].notna().values
    df = df[mask].reset_index(drop=True)
    embeddings = embeddings[mask]
    print(f"  Contrast filter: {mask.sum():,} / {len(mask):,} images kept "
          f"({df['contrast_type'].value_counts().to_dict()})")

    # ── Dataset / subject / slice summary ──
    print("\nDataset summary:")
    summary = (
        df.groupby("dataset")
        .agg(n_slices=("path", "count"),
             n_subjects=("subject", "nunique"))
        .sort_values("n_slices", ascending=False)
    )
    print(summary.to_string())
    summary.to_csv(output_dir / "summary_datasets.csv")

    lines = [f"{ds}  ({summary.loc[ds,'n_slices']:,} slices, {summary.loc[ds,'n_subjects']:,} subjects)"
             for ds in summary.index]
    (output_dir / "datasets_used.txt").write_text("\n".join(lines) + "\n")

    contrast_type_dist = df["contrast_type"].value_counts().reset_index()
    contrast_type_dist.columns = ["contrast_type", "count"]
    print(f"\nContrast type distribution (after filter):")
    print(contrast_type_dist.to_string(index=False))
    contrast_type_dist.to_csv(output_dir / "summary_contrasts.csv", index=False)

    probe_cols = ["contrast_type", "body_part", "pathology",
                  "dataset", "spacing_bin"]

    # ── Train / holdout split ──
    holdout_emb, holdout_df = None, None
    train_emb, train_df = embeddings, df
    if args.holdout_datasets:
        holdout_names = {d.strip() for d in args.holdout_datasets.split(",")}
        holdout_mask = df["dataset"].isin(holdout_names).values
        holdout_emb = embeddings[holdout_mask]
        holdout_df  = df[holdout_mask].reset_index(drop=True)
        train_emb   = embeddings[~holdout_mask]
        train_df    = df[~holdout_mask].reset_index(drop=True)
        print(f"\nHoldout set ({holdout_mask.sum():,} images): {sorted(holdout_names)}")
        print(f"Train set   ({(~holdout_mask).sum():,} images)")

    # ── Linear probing ──
    if not args.no_linear_probing:
        print("Running linear probing…")
        results = linear_probing(train_emb, train_df, probe_cols, output_dir,
                                 holdout_emb=holdout_emb, holdout_df=holdout_df)
        results.to_csv(output_dir / "linear_probing_results.csv", index=False)
        plot_linear_probing(results, output_dir)
        drop_cols = [c for c in ["classes"] if c in results.columns]
        print("\n" + results.drop(columns=drop_cols).to_string(index=False))

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
