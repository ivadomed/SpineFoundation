#!/usr/bin/env python3
"""
Patch-level MS lesion detection probing on Curia spatial features.

Uses seg_feat_cache/Curia_features.npy (N, 768, 32, 32) float16 memmap.
Binary label per patch cell (16×16px): lesion present or not.

Training set per fold (nnUNet-inspired):
  - ALL positive patches from foreground slices   (~17k patches, 54 MB)
  - 10× negative patches from background slices  (~175k patches, 540 MB)
  → Total ~600 MB → LogisticRegression directe, pas de partial_fit.

Evaluation on the full val set (natural distribution).
5-fold group CV (held-out subjects). Metric: balanced accuracy.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Config ─────────────────────────────────────────────────────────────────────
FEAT_PATH  = Path("seg_feat_cache/Curia_features.npy")
PATHS_FILE = Path("seg_feat_cache/Curia_paths.txt")
LABEL_ROOT = Path("/home/ge.polymtl.ca/p123239/data_work/01_extracted_v2/label_lesion")
DATASET    = "nih-ms-mp2rage"
ORIENT     = "axial"          # filter: only keep this orientation (None = all)
GRID       = 32
FEAT_DIM   = 768
N_FOLDS    = 5
NEG_RATIO  = 10     # negative patches per positive patch
VAL_BATCH  = 500    # slices per val batch (memmap read)

# ── Path utilities ──────────────────────────────────────────────────────────────

def path_to_key(path_str: str) -> str | None:
    stem  = Path(path_str).stem
    parts = stem.split("__")
    if len(parts) < 6:
        return None
    subject, _, orientation, sidx, tidx, spacing = parts[:6]
    if not re.match(r"s\d+", sidx):
        return None
    return f"{subject}__{orientation}__{sidx}__{tidx}__{spacing}"


def build_label_path_index() -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for split in ("train", "val"):
        d = LABEL_ROOT / split / DATASET
        if d.exists():
            for p in d.glob("*.png"):
                k = path_to_key(str(p))
                if k and k not in idx:
                    idx[k] = p
    return idx


def get_subject(path: str) -> str:
    return Path(path).stem.split("__")[0]


# ── Precompute patch labels ─────────────────────────────────────────────────────

def compute_patch_labels(img_paths: list[str],
                         label_index: dict[str, Path]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.zeros((len(img_paths), GRID * GRID), dtype=np.uint8)
    valid  = np.zeros(len(img_paths), dtype=bool)
    for i, ip in enumerate(tqdm(img_paths, desc="Computing patch labels")):
        k  = path_to_key(ip)
        lp = label_index.get(k)
        if lp is None:
            continue
        mask       = np.array(Image.open(lp).convert("L"))
        mask_small = np.array(Image.fromarray(mask).resize((GRID, GRID), Image.LANCZOS))
        labels[i]  = (mask_small > 0).flatten().astype(np.uint8)
        valid[i]   = True
    n_valid     = int(valid.sum())
    pos_patches = int(labels[valid].sum())
    tot_patches = n_valid * GRID * GRID
    print(f"  Valid slices    : {n_valid}/{len(img_paths)}")
    print(f"  Positive patches: {pos_patches}/{tot_patches}  ({100*pos_patches/tot_patches:.3f}%)")
    return labels, valid


# ── Load patches from slice indices ─────────────────────────────────────────────

def load_patches(mm: np.memmap, patch_labels: np.ndarray,
                 indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx   = np.sort(indices)
    feats = mm[idx].transpose(0, 2, 3, 1).reshape(-1, FEAT_DIM).astype(np.float32)
    lbls  = patch_labels[idx].reshape(-1)
    return feats, lbls


# ── Build training set ──────────────────────────────────────────────────────────

def build_train_set(mm: np.memmap, patch_labels: np.ndarray,
                    tr_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    nnUNet-inspired: all positive patches + NEG_RATIO × negatives.
    Negatives come from BOTH fg slices (natural ~10%) and bg slices,
    so the model can't use slice-level context as a shortcut.
    Fg slices are loaded in chunks to avoid peak RAM issues.
    """
    is_fg  = patch_labels[tr_idx].sum(axis=1) > 0
    fg_idx = tr_idx[is_fg]
    bg_idx = tr_idx[~is_fg]

    # ── Load fg slices in chunks: collect all positives + some negatives ──────
    # ~10% of negatives come from fg slices (matches natural distribution)
    NEG_PER_FG_SLICE = 8   # negative patches sampled per fg slice
    CHUNK = 300

    X_pos_list, X_neg_fg_list = [], []
    print(f"  Loading {len(fg_idx)} fg slices in chunks…")
    for start in range(0, len(fg_idx), CHUNK):
        chunk   = fg_idx[start:start + CHUNK]
        X_c, y_c = load_patches(mm, patch_labels, chunk)
        X_pos_list.append(X_c[y_c == 1].copy())
        X_neg_c = X_c[y_c == 0]
        n_sample = min(len(X_neg_c), len(chunk) * NEG_PER_FG_SLICE)
        perm     = np.random.permutation(len(X_neg_c))[:n_sample]
        X_neg_fg_list.append(X_neg_c[perm].copy())
        del X_c, y_c, X_neg_c

    X_pos    = np.vstack(X_pos_list);    del X_pos_list
    X_neg_fg = np.vstack(X_neg_fg_list); del X_neg_fg_list

    # ── Remaining negatives from bg slices ────────────────────────────────────
    n_neg_total  = len(X_pos) * NEG_RATIO
    n_neg_bg     = max(0, n_neg_total - len(X_neg_fg))
    n_neg_slices = min(n_neg_bg // (GRID * GRID) + 1, len(bg_idx))
    neg_sample   = np.random.choice(bg_idx, n_neg_slices, replace=False)
    print(f"  Loading {n_neg_slices} bg slices for {n_neg_bg} negatives…")
    X_bg, _ = load_patches(mm, patch_labels, neg_sample)
    perm     = np.random.permutation(len(X_bg))[:n_neg_bg]
    X_neg_bg = X_bg[perm].copy(); del X_bg

    X_neg = np.vstack([X_neg_fg, X_neg_bg])
    del X_neg_fg, X_neg_bg

    X_tr = np.vstack([X_pos, X_neg])
    y_tr = np.concatenate([np.ones(len(X_pos), dtype=np.uint8),
                           np.zeros(len(X_neg), dtype=np.uint8)])
    print(f"  Train: {len(X_tr)} patches  "
          f"({len(X_pos)} pos / {len(X_neg)} neg)  "
          f"{X_tr.nbytes/1e9:.2f} GB")
    return X_tr, y_tr


# ── Probe ───────────────────────────────────────────────────────────────────────

def visualize_examples(mm: np.memmap, patch_labels: np.ndarray,
                       img_paths: list[str], va_idx: np.ndarray,
                       model, scaler, n_examples: int = 8,
                       out_path: str = "patch_probing_examples.png") -> None:
    """Show TP/FP/FN/TN overlays on original slices for val slices with lesions."""
    INPUT_SIZE = 512
    CELL = INPUT_SIZE // GRID   # 16px per cell

    # Select val slices that have at least one positive patch
    pos_val = va_idx[patch_labels[va_idx].sum(axis=1) > 0]
    sample  = np.random.choice(pos_val, min(n_examples, len(pos_val)), replace=False)

    fig, axes = plt.subplots(len(sample), 3, figsize=(12, 4 * len(sample)))
    if len(sample) == 1:
        axes = axes[np.newaxis]

    for row, sl in enumerate(sample):
        # Original image
        img = np.array(Image.open(img_paths[sl]).convert("L").resize(
            (INPUT_SIZE, INPUT_SIZE), Image.LANCZOS))

        # Predict
        X_sl  = mm[sl].transpose(1, 2, 0).reshape(-1, FEAT_DIM).astype(np.float32)
        X_sl  = scaler.transform(X_sl)
        y_pred = model.predict(X_sl).reshape(GRID, GRID)
        y_true = patch_labels[sl].reshape(GRID, GRID)

        TP = (y_pred == 1) & (y_true == 1)
        FP = (y_pred == 1) & (y_true == 0)
        FN = (y_pred == 0) & (y_true == 1)

        def make_overlay(mask_dict):
            overlay = np.zeros((INPUT_SIZE, INPUT_SIZE, 4), dtype=np.float32)
            colors  = {"TP": (0, 1, 0, 0.55), "FP": (1, 0, 0, 0.55),
                       "FN": (1, 0.5, 0, 0.55)}
            for name, (r, g, b, a) in colors.items():
                m = mask_dict[name]
                for i in range(GRID):
                    for j in range(GRID):
                        if m[i, j]:
                            overlay[i*CELL:(i+1)*CELL, j*CELL:(j+1)*CELL] = (r, g, b, a)
            return overlay

        gt_overlay = make_overlay({"TP": y_true, "FP": np.zeros_like(y_true), "FN": np.zeros_like(y_true)})
        pr_overlay = make_overlay({"TP": TP, "FP": FP, "FN": FN})

        subj  = Path(img_paths[sl]).stem.split("__")[0]
        sidx  = Path(img_paths[sl]).stem.split("__")[3]
        n_pos = int(y_true.sum())

        axes[row, 0].imshow(img, cmap="gray"); axes[row, 0].axis("off")
        axes[row, 0].set_title(f"{subj} {sidx}  ({n_pos} lesion patches)", fontsize=8)

        axes[row, 1].imshow(img, cmap="gray")
        axes[row, 1].imshow(gt_overlay); axes[row, 1].axis("off")
        axes[row, 1].set_title("Ground truth (green)", fontsize=8)

        axes[row, 2].imshow(img, cmap="gray")
        axes[row, 2].imshow(pr_overlay); axes[row, 2].axis("off")
        axes[row, 2].set_title("TP=green  FP=red  FN=orange", fontsize=8)

    legend = [mpatches.Patch(color="green",  label="TP"),
              mpatches.Patch(color="red",    label="FP"),
              mpatches.Patch(color="orange", label="FN")]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=9)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  Saved {out_path}")


def run_fold(mm: np.memmap, patch_labels: np.ndarray,
             tr_idx: np.ndarray, va_idx: np.ndarray,
             fold: int, img_paths: list[str] | None = None) -> float:

    X_tr, y_tr = build_train_set(mm, patch_labels, tr_idx)

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)

    print("  Fitting LogisticRegression…")
    model = LogisticRegression(class_weight="balanced", solver="saga",
                               max_iter=500, C=1.0, tol=1e-3)
    model.fit(X_tr, y_tr)
    del X_tr, y_tr

    # ── Evaluate on full val set (natural distribution) ──────────────────────
    all_pred, all_true = [], []
    va_sorted = np.sort(va_idx)
    for start in tqdm(range(0, len(va_sorted), VAL_BATCH),
                      desc=f"  fold {fold+1} val", leave=False):
        bidx     = va_sorted[start:start + VAL_BATCH]
        X_b, y_b = load_patches(mm, patch_labels, bidx)
        X_b      = scaler.transform(X_b)
        all_pred.append(model.predict(X_b))
        all_true.append(y_b)
        del X_b

    y_pred  = np.concatenate(all_pred)
    y_true  = np.concatenate(all_true)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    print(f"  fold {fold+1} → balanced_acc={bal_acc:.4f}")

    if fold == 0 and img_paths is not None:
        print("  Generating examples visualization…")
        visualize_examples(mm, patch_labels, img_paths, va_idx, model, scaler)

    return bal_acc


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    img_paths   = PATHS_FILE.read_text().strip().split("\n")
    label_index = build_label_path_index()

    n_slices  = FEAT_PATH.stat().st_size // (FEAT_DIM * GRID * GRID * 2)
    img_paths = img_paths[:n_slices]
    print(f"Memmap slices: {n_slices}  |  label index: {len(label_index)} keys")

    # Filter by orientation before computing labels
    if ORIENT:
        orient_mask = np.array(
            [Path(p).stem.split("__")[2] == ORIENT for p in img_paths], dtype=bool)
        print(f"Orientation filter '{ORIENT}': {orient_mask.sum()}/{len(img_paths)} slices")
    else:
        orient_mask = np.ones(len(img_paths), dtype=bool)

    patch_labels, valid = compute_patch_labels(img_paths, label_index)
    valid = valid & orient_mask
    valid_idx = np.where(valid)[0]
    subjects  = np.array([get_subject(img_paths[i]) for i in valid_idx])
    print(f"Subjects: {len(np.unique(subjects))}")

    mm = np.memmap(FEAT_PATH, dtype="float16", mode="r",
                   shape=(n_slices, FEAT_DIM, GRID, GRID))

    gkf    = GroupKFold(n_splits=N_FOLDS)
    scores = []

    for fold, (tr, va) in enumerate(gkf.split(valid_idx, groups=subjects)):
        print(f"\n── Fold {fold+1}/{N_FOLDS}  train={len(tr)}  val={len(va)} ──")
        bal_acc = run_fold(mm, patch_labels, valid_idx[tr], valid_idx[va], fold,
                          img_paths=img_paths)
        scores.append(bal_acc)

    print(f"\n{'='*55}")
    print(f"Patch-level lesion detection — Curia ({GRID}×{GRID} grid)")
    print(f"  Balanced acc = {np.mean(scores):.4f} ± {np.std(scores):.4f}  {scores}")


if __name__ == "__main__":
    main()
