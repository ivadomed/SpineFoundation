"""
analyze_label_probing.py

Probes foundation model CLS embeddings against mask labels (cord seg or lesion).
Reuses the embedding cache from analyze_embeddings.py.

Modes
-----
  cord_seg  (default)
      Labels = spinal cord segmentation masks (label/).
      Probes: CSA regression, cord-visible binary, CSA-bin classification.
      Slices with mask=0 are OFF the cord → filtered out of regression/bin probes.

  lesion
      Labels = MS lesion masks (label_lesion/).
      Probes: lesion-present binary, lesion-load regression (lesion+ slices),
              lesion-load bin, per-dataset lesion prevalence.
      Slices with mask=0 are TRUE NEGATIVES (no lesion at that level) → kept
      for the binary probe, filtered only for load regression.

Signal comparison: CLS token vs mean-pool of patch tokens (cached separately).

Usage — cord seg
-----
python analyze_label_probing.py \\
    --data_dir  /path/to/01_extracted_v2 \\
    --model     models/curia \\
    --output_dir ./analysis_output_label_cord \\
    --cache_dir  ./analysis_output \\
    --orientation axial --datasets canproco,nih-ms-mp2rage,data-multi-subject

Usage — lesion
-----
python analyze_label_probing.py \\
    --data_dir  /path/to/01_extracted_v2 \\
    --label_dir /path/to/01_extracted_v2/label_lesion \\
    --mode      lesion \\
    --model     models/curia \\
    --output_dir ./analysis_output_label_lesion \\
    --cache_dir  ./analysis_output \\
    --orientation axial \\
    --datasets   canproco,nih-ms-mp2rage,ms-mayo-critical-lesions-2025
"""

from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, KFold, GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, balanced_accuracy_score, f1_score
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


# ─── Helpers shared with analyze_embeddings.py ───────────────────────────────

def _slug(path: str) -> str:
    return os.path.normpath(path).replace("/", "_").replace("\\", "_").lstrip("_.")


def _parse_metadata(filename: str) -> dict:
    """Re-parse the same metadata fields as analyze_embeddings.py."""
    stem = Path(filename).stem
    parts = stem.split("__")
    subject = parts[0] if len(parts) > 0 else "unknown"
    contrast = "unknown"
    orientation = "unknown"
    slice_idx = 0
    spacing = (1.0, 1.0)

    if len(parts) >= 2:
        raw_contrast = parts[1].split("_")[-1] if "_" in parts[1] else parts[1]
        m = re.search(r'(?:^|_)(T1w|T2w|T2star|FLAIR|PDw|MTR|MTS|MT|DWI|'
                      r'MP2RAGE|UNIT1|PSIR|STIR|phase|T1map|T2map)(?:$|_)',
                      raw_contrast, re.IGNORECASE)
        if m:
            contrast = m.group(1).upper()
            contrast = {"T2STAR": "T2star", "UNIT1": "UNIT1",
                        "MP2RAGE": "MP2RAGE"}.get(contrast, contrast.capitalize())

    for p in parts:
        if p in ("axial", "sagittal", "coronal"):
            orientation = p
        m = re.match(r"s(\d+)", p)
        if m:
            slice_idx = int(m.group(1))
        m = re.match(r"sp(\d+)x(\d+)", p)
        if m:
            spacing = (int(m.group(1)) / 1000.0, int(m.group(2)) / 1000.0)

    return dict(subject=subject, contrast_raw=contrast,
                orientation=orientation, slice_idx=slice_idx, spacing=spacing)


# ─── Model loading ─────────────────────────────────────────────────────────--

def load_model(model_name: str, device: torch.device):
    print(f"Loading model: {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
    model = torch.compile(model)
    return model, processor


# ─── Dataset / DataLoader ────────────────────────────────────────────────────

class _ImageDataset(Dataset):
    def __init__(self, paths: list[str], img_mode: str):
        self.paths = paths
        self.img_mode = img_mode

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return np.array(Image.open(self.paths[idx]).convert(self.img_mode))


# ─── Embedding extraction (CLS + patch mean) ─────────────────────────────────

@torch.no_grad()
def extract_embeddings(df: pd.DataFrame, model, processor,
                       device: torch.device, batch_size: int,
                       num_workers: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Returns (cls_embeddings, patch_mean_embeddings), both shape (N, D)."""
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    img_mode = "L" if getattr(inner.config, "num_channels", 3) == 1 else "RGB"

    def collate(images):
        inputs = processor(images=images, return_tensors="pt")
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
    cls_list: list[np.ndarray] = []
    patch_list: list[np.ndarray] = []

    for inputs in tqdm(loader, desc="Extracting CLS + patch-mean"):
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
            out = model(**inputs)
        hs = out.last_hidden_state.float()   # (B, 1+N_patches, D)
        cls_list.append(hs[:, 0, :].cpu().numpy())
        patch_list.append(hs[:, 1:, :].mean(dim=1).cpu().numpy())

    return np.vstack(cls_list), np.vstack(patch_list)


# ─── Cord segmentation mask loading ──────────────────────────────────────────

# Module-level worker (required for ProcessPoolExecutor pickling)
def _count_pixels(mask_path_str: str) -> int:
    p = Path(mask_path_str)
    if not p.exists():
        return 0
    return int((np.array(Image.open(p)) > 0).sum())


def load_cord_masks(df: pd.DataFrame, label_dir: Path,
                    cache_path: Path | None = None,
                    recompute: bool = False,
                    workers: int = 16) -> np.ndarray:
    """
    For each row in df, count foreground pixels in the matching mask PNG.
    Returns array of shape (N,) with integer pixel counts (0 if no mask found).
    Parallelised with ProcessPoolExecutor; result cached to cache_path if given.
    """
    if cache_path is not None and cache_path.exists() and not recompute:
        print(f"  Loading pixel-count cache from {cache_path.name}")
        return np.load(cache_path)

    label_dir = Path(label_dir)
    paths = [
        str(label_dir / (row.split if hasattr(row, "split") else "train")
            / row.dataset / Path(row.path).name)
        for row in df.itertuples()
    ]

    counts = np.zeros(len(paths), dtype=np.int32)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_count_pixels, p): i for i, p in enumerate(paths)}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Loading masks"):
            counts[futs[fut]] = fut.result()

    if cache_path is not None:
        np.save(cache_path, counts)
        print(f"  Saved pixel-count cache to {cache_path.name}")
    return counts


# ─── Regression probe ────────────────────────────────────────────────────────

def probe_regression(X: np.ndarray, y: np.ndarray,
                     groups: np.ndarray | None = None,
                     n_splits: int = 5) -> dict:
    """5-fold linear (Ridge) regression. Returns R², Pearson r, MSE (mean±std)."""
    if groups is not None and len(set(groups)) >= n_splits:
        cv = GroupKFold(n_splits=n_splits)
        split_iter = cv.split(X, y, groups=groups)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_iter = cv.split(X, y)

    r2s, rs, mses = [], [], []
    for train_idx, val_idx in split_iter:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        reg = Ridge(alpha=1.0)
        reg.fit(X_tr, y[train_idx])
        pred = reg.predict(X_val)
        r2s.append(r2_score(y[val_idx], pred))
        rs.append(pearsonr(y[val_idx], pred)[0])
        mses.append(float(np.mean((y[val_idx] - pred) ** 2)))

    return dict(
        r2_mean=float(np.mean(r2s)),   r2_std=float(np.std(r2s)),
        r_mean=float(np.mean(rs)),     r_std=float(np.std(rs)),
        mse_mean=float(np.mean(mses)), mse_std=float(np.std(mses)),
    )


# ─── Classification probe (re-used from analyze_embeddings.py style) ─────────

def probe_classification(X: np.ndarray, y: np.ndarray,
                         groups: np.ndarray | None = None,
                         n_splits: int = 5) -> dict:
    """5-fold logistic regression. Returns balanced accuracy mean±std."""
    n_classes = len(np.unique(y))
    use_group = groups is not None and len(set(groups)) >= n_splits

    if use_group:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_iter = cv.split(X, y, groups=groups)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_iter = cv.split(X, y)

    bal_accs, f1s = [], []
    for train_idx, val_idx in split_iter:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
                                 solver="saga", n_jobs=-1)
        clf.fit(X_tr, y[train_idx])
        pred = clf.predict(X_val)
        bal_accs.append(balanced_accuracy_score(y[val_idx], pred))
        f1s.append(f1_score(y[val_idx], pred, average="macro"))

    return dict(
        bal_acc_mean=float(np.mean(bal_accs)), bal_acc_std=float(np.std(bal_accs)),
        f1_mean=float(np.mean(f1s)),           f1_std=float(np.std(f1s)),
        chance=float(1.0 / n_classes),
    )


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_regression_comparison(results: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart: CLS vs patch-mean on R² and Pearson r for each probe."""
    probes = results["probe"].unique()
    n = len(probes)
    x = np.arange(n)
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(max(8, n * 2.5), 5))
    colors = {"CLS": "#4C72B0", "patch_mean": "#DD8452"}

    for ax, metric, ylabel in zip(axes,
                                   ["r2_mean", "r_mean"],
                                   ["R²  (5-fold CV)", "Pearson  r  (5-fold CV)"]):
        for i, (token, offset) in enumerate([("CLS", -w/2), ("patch_mean", w/2)]):
            sub = results[results["token"] == token].set_index("probe")
            vals  = [sub.loc[p, metric] if p in sub.index else 0 for p in probes]
            stds  = [sub.loc[p, metric.replace("mean", "std")] if p in sub.index else 0 for p in probes]
            ax.bar(x + offset, vals, w, yerr=stds, label=token,
                   color=colors[token], capsize=4, alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(probes, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.axhline(0, color="k", linewidth=0.5)

    plt.suptitle("CLS token vs patch-mean — regression probes", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "label_probing_regression.png", dpi=150)
    plt.close()


def plot_classification_comparison(results: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart: CLS vs patch-mean on balanced accuracy for classification probes."""
    probes = results["probe"].unique()
    n = len(probes)
    x = np.arange(n)
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(8, n * 2.5), 5))
    colors = {"CLS": "#4C72B0", "patch_mean": "#DD8452"}

    for token, offset in [("CLS", -w/2), ("patch_mean", w/2)]:
        sub = results[results["token"] == token].set_index("probe")
        vals  = [sub.loc[p, "bal_acc_mean"] if p in sub.index else 0 for p in probes]
        stds  = [sub.loc[p, "bal_acc_std"]  if p in sub.index else 0 for p in probes]
        chances = [sub.loc[p, "chance"] if p in sub.index else 0 for p in probes]
        ax.bar(x + offset, vals, w, yerr=stds, label=token,
               color=colors[token], capsize=4, alpha=0.9)

    # Chance levels
    for i, p in enumerate(probes):
        sub = results[(results["probe"] == p) & (results["token"] == "CLS")]
        if not sub.empty:
            ax.plot([i - w, i + w], [sub.iloc[0]["chance"]] * 2, "r--", linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(probes, rotation=15, ha="right")
    ax.set_ylabel("Balanced accuracy (5-fold CV)")
    ax.set_title("CLS token vs patch-mean — classification probes", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "label_probing_classification.png", dpi=150)
    plt.close()


def plot_csa_scatter(csa: np.ndarray, cls_preds: np.ndarray,
                     patch_preds: np.ndarray, dataset: pd.Series,
                     output_dir: Path) -> None:
    """Scatter: true vs predicted CSA for CLS and patch-mean on a random fold."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ds_labels = dataset.unique()
    cmap = plt.get_cmap("tab10")
    color_map = {ds: cmap(i % 10) for i, ds in enumerate(sorted(ds_labels))}
    colors = [color_map[d] for d in dataset]

    for ax, preds, title in [(axes[0], cls_preds, "CLS token"),
                              (axes[1], patch_preds, "Patch-mean")]:
        ax.scatter(csa, preds, c=colors, s=3, alpha=0.4, rasterized=True)
        lo, hi = min(csa.min(), preds.min()), max(csa.max(), preds.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        r2 = r2_score(csa, preds)
        r, _ = pearsonr(csa, preds)
        ax.set_title(f"{title}  (R²={r2:.3f}, r={r:.3f})")
        ax.set_xlabel("True CSA (pixels)")
        ax.set_ylabel("Predicted CSA (pixels)")

    patches = [mpatches.Patch(color=color_map[ds], label=ds) for ds in sorted(ds_labels)]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.05))
    plt.suptitle("CSA prediction: true vs predicted", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "label_probing_csa_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir",    required=True,
                        help="Root of 01_extracted_v2 (contains image/ and label/)")
    parser.add_argument("--label_dir",   default=None,
                        help="Root of label directory. Defaults to data_dir/label "
                             "for cord_seg mode, data_dir/label_lesion for lesion mode.")
    parser.add_argument("--mode",        default="cord_seg",
                        choices=["cord_seg", "lesion"],
                        help="cord_seg: CSA probes.  lesion: lesion-presence/load probes.")
    parser.add_argument("--model",       required=True)
    parser.add_argument("--output_dir",  default="./analysis_output_label")
    parser.add_argument("--cache_dir",   default=None,
                        help="Dir containing embedding cache from analyze_embeddings.py. "
                             "Defaults to --output_dir.")
    parser.add_argument("--split",       default="train", choices=["train", "val", "both"])
    parser.add_argument("--orientation", default=None, choices=["axial", "sagittal", "coronal"])
    parser.add_argument("--datasets",    default=None,
                        help="Comma-separated list of datasets to include.")
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--n_splits",    type=int, default=5)
    parser.add_argument("--recompute",   action="store_true",
                        help="Force re-extraction of patch-mean embeddings.")
    parser.add_argument("--no_patch_mean", action="store_true",
                        help="Skip patch-mean extraction (CLS-only comparison).")
    parser.add_argument("--device",      default=None)
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir
    data_dir  = Path(args.data_dir)
    mode = args.mode
    if args.label_dir:
        label_dir = Path(args.label_dir)
    else:
        label_dir = data_dir / ("label_lesion" if mode == "lesion" else "label")

    slug = _slug(args.model)
    orient_slug = f"_{args.orientation}" if args.orientation else ""

    # ── Load CLS embedding cache ──────────────────────────────────────────────
    cache_emb  = cache_dir / f"embeddings_{slug}{orient_slug}.npy"
    cache_meta = cache_dir / f"metadata_{slug}{orient_slug}.csv"

    if not cache_emb.exists() or not cache_meta.exists():
        raise FileNotFoundError(
            f"CLS embedding cache not found in {cache_dir}.\n"
            f"Run analyze_embeddings.py first with --cache_dir {cache_dir}"
        )

    print(f"Loading CLS embeddings from {cache_emb}")
    cls_emb = np.load(cache_emb)
    df = pd.read_csv(cache_meta)
    print(f"  {len(df):,} slices loaded")

    # ── Filters ───────────────────────────────────────────────────────────────
    if args.orientation:
        filtered = df[df["orientation"] == args.orientation]
        orig_idx = filtered.index.tolist()   # capture before reset_index
        cls_emb  = cls_emb[orig_idx]
        df = filtered.reset_index(drop=True)

    if args.datasets:
        wanted = {d.strip() for d in args.datasets.split(",")}
        mask = df["dataset"].isin(wanted).values
        df     = df[mask].reset_index(drop=True)
        cls_emb = cls_emb[mask]

    if args.split != "both":
        mask = (df["split"] == args.split).values
        df     = df[mask].reset_index(drop=True)
        cls_emb = cls_emb[mask]

    print(f"  After filters: {len(df):,} slices, {df['dataset'].nunique()} datasets")

    # ── Patch-mean embeddings ─────────────────────────────────────────────────
    cache_patch = cache_dir / f"patch_mean_{slug}{orient_slug}.npy"

    if args.no_patch_mean:
        patch_emb = None
        print("Skipping patch-mean extraction (--no_patch_mean)")
    elif cache_patch.exists() and not args.recompute:
        print(f"Loading patch-mean embeddings from {cache_patch}")
        patch_emb_full = np.load(cache_patch)
        # Re-apply same filters using the already-loaded full metadata
        df_full = pd.read_csv(cache_meta)
        mask_full = np.ones(len(df_full), dtype=bool)
        if args.orientation:
            mask_full &= df_full["orientation"].values == args.orientation
        if args.datasets:
            wanted = {d.strip() for d in args.datasets.split(",")}
            mask_full &= df_full["dataset"].isin(wanted).values
        if args.split != "both":
            mask_full &= df_full["split"].values == args.split
        patch_emb = patch_emb_full[mask_full]
        del patch_emb_full
    else:
        print("Extracting patch-mean embeddings (new forward pass)…")
        model, processor = load_model(args.model, device)
        cls_new, patch_emb_new = extract_embeddings(
            df, model, processor, device, args.batch_size, args.num_workers
        )
        # save full patch-mean cache (filtered subset)
        np.save(cache_patch, patch_emb_new)
        print(f"  Saved patch-mean cache: {cache_patch}")
        patch_emb = patch_emb_new
        del model

    # ── Load masks ────────────────────────────────────────────────────────────
    label_type = "lesion" if mode == "lesion" else "cord segmentation"
    print(f"Loading {label_type} masks…")
    pixel_cache = cache_dir / f"pixel_counts_{mode}_{slug}{orient_slug}.npy"
    pixel_count = load_cord_masks(df, label_dir,
                                  cache_path=pixel_cache,
                                  recompute=args.recompute,
                                  workers=16)
    has_mask = (pixel_count > 0)
    n_with  = has_mask.sum()
    n_total = len(df)
    print(f"  {n_with:,}/{n_total:,} slices have mask  "
          f"({'lesion+' if mode == 'lesion' else 'cord visible'})")

    if n_with < 50:
        raise RuntimeError(f"Too few slices with masks ({n_with}) — check --label_dir.")

    # Group key for CV (no subject leakage)
    has_subj = "subject" in df.columns
    groups = (df["dataset"] + "__" + df["subject"]).values if has_subj else None
    print(f"  CV: {'StratifiedGroupKFold' if groups is not None else 'KFold'} "
          f"({args.n_splits} splits)")

    tokens = {"CLS": cls_emb}
    if patch_emb is not None:
        tokens["patch_mean"] = patch_emb

    reg_rows: list[dict] = []
    clf_rows: list[dict] = []
    ds_rows:  list[dict] = []

    # ── Probes shared by both modes ───────────────────────────────────────────

    # Binary: mask present vs absent
    bin_label  = "lesion_present" if mode == "lesion" else "cord_visible"
    y_bin_all  = has_mask.astype(int)
    print(f"\n── Probe: {bin_label} (binary) ──")
    print(f"  positives={has_mask.sum():,}  negatives={(~has_mask).sum():,}")
    for name, emb in tokens.items():
        res = probe_classification(emb, y_bin_all, groups=groups, n_splits=args.n_splits)
        print(f"  {name:12s}  bal_acc={res['bal_acc_mean']:.3f}±{res['bal_acc_std']:.3f}  "
              f"chance={res['chance']:.3f}")
        clf_rows.append({"probe": bin_label, "token": name, **res})

    # Regression: pixel count (on masked slices only)
    reg_label = "lesion_load" if mode == "lesion" else "CSA"
    y_reg     = pixel_count[has_mask].astype(float)
    grp_reg   = groups[has_mask] if groups is not None else None
    print(f"\n── Probe: {reg_label} regression (n={has_mask.sum():,}) ──")
    for name, emb in tokens.items():
        X = emb[has_mask]
        res = probe_regression(X, y_reg, groups=grp_reg, n_splits=args.n_splits)
        print(f"  {name:12s}  R²={res['r2_mean']:.3f}±{res['r2_std']:.3f}  "
              f"r={res['r_mean']:.3f}±{res['r_std']:.3f}")
        reg_rows.append({"probe": reg_label, "token": name, **res})

    # Bin classification: small / medium / large load (on masked slices only)
    bin3_label = "lesion_load_bin" if mode == "lesion" else "CSA_bin"
    vals = pixel_count[has_mask]
    q33, q66 = np.percentile(vals, [33, 66])
    def to_bin(v):
        if v <= q33: return "small"
        if v <= q66: return "medium"
        return "large"
    y_bin3 = np.array([to_bin(v) for v in vals])
    le3 = LabelEncoder()
    y_bin3_enc = le3.fit_transform(y_bin3)
    print(f"\n── Probe: {bin3_label} (3-class) ──")
    for name, emb in tokens.items():
        X = emb[has_mask]
        res = probe_classification(X, y_bin3_enc, groups=grp_reg, n_splits=args.n_splits)
        print(f"  {name:12s}  bal_acc={res['bal_acc_mean']:.3f}±{res['bal_acc_std']:.3f}  "
              f"chance={res['chance']:.3f}")
        clf_rows.append({"probe": bin3_label, "token": name, **res})

    # ── Lesion-mode extra: per-dataset lesion prevalence ──────────────────────
    print(f"\n── Per-dataset breakdown ──")
    for ds in sorted(df["dataset"].unique()):
        mask_ds  = df["dataset"].values == ds
        n_ds     = mask_ds.sum()
        n_pos_ds = (has_mask & mask_ds).sum()
        prev     = n_pos_ds / n_ds if n_ds > 0 else 0.0

        # Per-dataset binary probe (CLS only)
        if n_pos_ds >= 20 and (n_ds - n_pos_ds) >= 20:
            grp_ds = groups[mask_ds] if groups is not None else None
            res = probe_classification(cls_emb[mask_ds], y_bin_all[mask_ds],
                                       groups=grp_ds, n_splits=min(args.n_splits, 3))
            bal_acc = res["bal_acc_mean"]
        else:
            bal_acc = float("nan")

        print(f"  {ds:35s}  n={n_ds:,}  lesion+={n_pos_ds:,} ({prev*100:.1f}%)  "
              f"bal_acc={bal_acc:.3f}" if not np.isnan(bal_acc) else
              f"  {ds:35s}  n={n_ds:,}  lesion+={n_pos_ds:,} ({prev*100:.1f}%)  bal_acc=n/a")
        ds_rows.append({"dataset": ds, "n_slices": int(n_ds),
                        "n_lesion_pos": int(n_pos_ds),
                        "lesion_prevalence": float(prev),
                        "bal_acc_binary": float(bal_acc)})

    # ── Scatter plot (regression, full-data fit) ───────────────────────────────
    scatter_preds = {}
    for name, emb in tokens.items():
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(emb[has_mask])
        scatter_preds[name] = Ridge(alpha=1.0).fit(X_sc, y_reg).predict(X_sc)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    pd.DataFrame(reg_rows).to_csv(output_dir / "label_probing_regression.csv", index=False)
    pd.DataFrame(clf_rows).to_csv(output_dir / "label_probing_classification.csv", index=False)
    pd.DataFrame(ds_rows).to_csv(output_dir  / "label_probing_per_dataset.csv",    index=False)
    print(f"\nCSVs saved to {output_dir}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_regression_comparison(pd.DataFrame(reg_rows), output_dir)
    plot_classification_comparison(pd.DataFrame(clf_rows), output_dir)

    patch_pred = scatter_preds.get("patch_mean", scatter_preds["CLS"])
    plot_csa_scatter(y_reg, scatter_preds["CLS"], patch_pred,
                     df.loc[has_mask, "dataset"].reset_index(drop=True), output_dir)

    # Per-dataset bar (lesion prevalence + bal_acc)
    ds_df = pd.DataFrame(ds_rows).dropna(subset=["bal_acc_binary"])
    if not ds_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(max(10, len(ds_df) * 1.4), 5))
        for ax, col, ylabel, color in [
            (axes[0], "lesion_prevalence", "Lesion prevalence (%)", "#DD8452"),
            (axes[1], "bal_acc_binary",    "Bal. acc — lesion present (CLS)", "#4C72B0"),
        ]:
            vals = ds_df[col] * (100 if col == "lesion_prevalence" else 1)
            ax.bar(range(len(ds_df)), vals, color=color, alpha=0.9)
            ax.set_xticks(range(len(ds_df)))
            ax.set_xticklabels(ds_df["dataset"], rotation=30, ha="right", fontsize=8)
            ax.set_ylabel(ylabel)
            if col == "bal_acc_binary":
                ax.axhline(0.5, color="r", linestyle="--", linewidth=1)
                ax.set_ylim(0, 1.05)
        plt.suptitle(f"Per-dataset lesion stats — {mode} mode", fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / "label_probing_per_dataset.png", dpi=150)
        plt.close()

    print("\nDone.")
    print(f"Outputs in: {output_dir}")


if __name__ == "__main__":
    main()
