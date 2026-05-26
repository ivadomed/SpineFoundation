#!/usr/bin/env python3
"""
Binary lesion-presence probing on NIH-MS-MP2RAGE.

For each foundation model, loads cached embeddings, matches slices
to their lesion-mask PNG (has_lesion = 1 if any pixel > 0), then
trains a logistic-regression probe with 5-fold stratified group CV
(held-out subjects).

Usage:
    python analyze_lesion_probing.py

Outputs:
    lesion_probing_results.csv   — one row per model
    lesion_probing_results.png   — bar chart
"""

from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
DATASET = "nih-ms-mp2rage"
LABEL_ROOT = Path("/home/ge.polymtl.ca/p123239/data_work/01_extracted_v2/label_lesion")

MODELS = {
    "Curia":        "analysis_output",
    "DINOv2+reg":   "analysis_output_dinov2",
    "DINOv3 ViT-L": "analysis_output_dinov3",
    "MRI-CORE":     "analysis_output_mricore",
}
ORIENT_SLUG = "_axial"
N_FOLDS = 5
OUT_CSV = Path("lesion_probing_results.csv")
OUT_PNG = Path("lesion_probing_results.png")


# ── Key extraction ────────────────────────────────────────────────────────────

def path_to_key(path_str: str) -> str | None:
    """Extract matching key from an image/label path.

    Key = subject__orientation__s####__t###__sp###
    (contrast-agnostic so image and label filenames match)
    """
    stem = Path(path_str).stem
    parts = stem.split("__")
    if len(parts) < 6:
        return None
    subject, _, orientation, sidx, tidx, spacing = parts[:6]
    if not re.match(r"s\d+", sidx):
        return None
    return f"{subject}__{orientation}__{sidx}__{tidx}__{spacing}"


# ── Build label index ─────────────────────────────────────────────────────────

def _label_value(png_path: Path) -> tuple[str, int]:
    key = path_to_key(str(png_path))
    has_lesion = int(np.array(Image.open(png_path)).max() > 0)
    return key, has_lesion


def build_label_index(dataset: str) -> dict[str, int]:
    """Return {key: has_lesion} for all label PNGs of a dataset."""
    paths = []
    for split in ("train", "val"):
        d = LABEL_ROOT / split / dataset
        if d.exists():
            paths.extend(d.glob("*.png"))

    print(f"Building label index from {len(paths)} PNGs …")
    index: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(_label_value, p): p for p in paths}
        for fut in tqdm(as_completed(futs), total=len(paths), unit="file"):
            key, val = fut.result()
            if key is not None:
                # if multiple contrasts map to same key, OR them (lesion present in any)
                index[key] = max(index.get(key, 0), val)

    n_pos = sum(v for v in index.values())
    print(f"  Keys: {len(index)}  |  Positive (has lesion): {n_pos} ({100*n_pos/len(index):.1f}%)")
    return index


# ── Linear probing ────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR        = 1e-3
EPOCHS    = 300
BATCH     = 2048
PATIENCE  = 20   # early stopping on train loss


def _train_probe(X_tr: np.ndarray, y_tr: np.ndarray,
                 X_va: np.ndarray, y_va: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score

    # z-score on CPU (fast, avoids GPU memory spike)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)

    X_tr_t = torch.from_numpy(X_tr).float().to(DEVICE)
    y_tr_t = torch.from_numpy(y_tr).float().to(DEVICE)
    X_va_t = torch.from_numpy(X_va).float().to(DEVICE)

    dim = X_tr_t.shape[1]
    linear = nn.Linear(dim, 1).to(DEVICE)

    # class-balanced pos_weight
    n_pos = y_tr.sum()
    n_neg = len(y_tr) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(linear.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    n = len(X_tr_t)
    best_loss = float("inf")
    epochs_no_improve = 0
    stopped_at = EPOCHS

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        perm = torch.randperm(n, device=DEVICE)
        for start in range(0, n, BATCH):
            idx = perm[start:start + BATCH]
            optimizer.zero_grad()
            loss = criterion(linear(X_tr_t[idx]).squeeze(), y_tr_t[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step(epoch_loss)

        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                stopped_at = epoch + 1
                break

    with torch.no_grad():
        logits = linear(X_va_t).squeeze().cpu().numpy()
    preds = (logits > 0).astype(int)
    acc = balanced_accuracy_score(y_va, preds)
    return acc, stopped_at


def probe(embeddings: np.ndarray, labels: np.ndarray,
          groups: np.ndarray) -> tuple[float, float, list[float]]:
    """5-fold stratified group CV on GPU → (mean_bal_acc, std_bal_acc, fold_scores)."""
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS)
    scores = []
    for fold, (tr, va) in enumerate(sgkf.split(embeddings, labels, groups)):
        acc, stopped_at = _train_probe(embeddings[tr], labels[tr], embeddings[va], labels[va])
        converged = stopped_at < EPOCHS
        print(f"    fold {fold+1}/{N_FOLDS}: bal_acc={acc:.3f}  stopped at epoch {stopped_at} ({'converged' if converged else 'MAX EPOCHS HIT'})")
        scores.append(acc)
    return float(np.mean(scores)), float(np.std(scores)), scores


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    label_index = build_label_index(DATASET)

    results = []
    for model_name, cache_dir in MODELS.items():
        cache = Path(cache_dir)
        # find embedding cache files
        npy_files = list(cache.glob(f"embeddings_*{ORIENT_SLUG}.npy"))
        if not npy_files:
            print(f"[{model_name}] No embedding cache found in {cache} — skipping")
            continue
        emb_path  = npy_files[0]
        meta_path = cache / emb_path.name.replace("embeddings_", "metadata_").replace(".npy", ".csv")

        print(f"\n[{model_name}] Loading {emb_path.name} …")
        emb_all  = np.load(emb_path)
        meta_all = pd.read_csv(meta_path)

        # filter to our dataset
        mask = meta_all["dataset"] == DATASET
        emb  = emb_all[mask]
        meta = meta_all[mask].reset_index(drop=True)
        print(f"  {len(meta)} slices for {DATASET}")

        # match with label index
        keys     = meta["path"].apply(path_to_key)
        has_lesion = keys.map(label_index)
        valid    = has_lesion.notna()
        print(f"  Matched: {valid.sum()} / {len(meta)} slices")

        emb    = emb[valid.values]
        labels = has_lesion[valid].astype(int).values
        groups = meta["subject"][valid].values

        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        majority_acc = max(n_pos, n_neg) / len(labels)
        print(f"  Positive: {n_pos} ({100*n_pos/len(labels):.1f}%)  Negative: {n_neg}  Majority-class acc: {majority_acc:.3f}  Bal-acc chance: 0.500")

        mean_acc, std_acc, fold_scores = probe(emb, labels, groups)
        print(f"  → Balanced acc: {mean_acc:.3f} ± {std_acc:.3f}")

        results.append({
            "model":           model_name,
            "n_slices":        len(labels),
            "n_positive":      int(n_pos),
            "prevalence":      round(n_pos / len(labels), 3),
            "majority_acc":    round(majority_acc, 3),
            "bal_acc_chance":  0.500,
            "bal_acc_mean":    round(mean_acc, 4),
            "bal_acc_std":     round(std_acc, 4),
            "fold_scores":     fold_scores,
        })

    from scipy.stats import ttest_rel

    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)

    # ── Paired t-tests vs best model ─────────────────────────────────────────
    best_idx = df["bal_acc_mean"].idxmax()
    best = df.iloc[best_idx]
    print(f"\nResults (best: {best['model']} = {best['bal_acc_mean']:.4f})")
    print(f"{'Model':<15} {'Mean':>7} {'Std':>7} {'vs best p-value':>16} {'significant':>12}")
    for _, row in df.iterrows():
        if row["model"] == best["model"]:
            print(f"  {row['model']:<13} {row['bal_acc_mean']:>7.4f} {row['bal_acc_std']:>7.4f} {'—':>16} {'(reference)':>12}")
        else:
            _, p = ttest_rel(best["fold_scores"], row["fold_scores"])
            sig = "YES *" if p < 0.05 else "no"
            print(f"  {row['model']:<13} {row['bal_acc_mean']:>7.4f} {row['bal_acc_std']:>7.4f} {p:>16.4f} {sig:>12}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2166ac", "#4dac26", "#d6604d", "#8073ac"]
    x = np.arange(len(df))
    bars = ax.bar(x, df["bal_acc_mean"], yerr=df["bal_acc_std"], capsize=5,
                  color=colors[:len(df)], alpha=0.85, width=0.5)

    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1.5, label="Chance (bal. acc. = 0.500)")

    for bar, row in zip(bars, df.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                row.bal_acc_mean + row.bal_acc_std + 0.008,
                f"{row.bal_acc_mean:.3f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], fontsize=10)
    ax.set_ylabel("Balanced accuracy (5-fold CV)", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"MS lesion presence detection — {DATASET}\n"
                 f"Binary linear probe ({N_FOLDS}-fold stratified group CV)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {OUT_PNG}")


if __name__ == "__main__":
    main()
