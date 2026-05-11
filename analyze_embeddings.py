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
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


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


def parse_filename(filename: str) -> dict:
    """Parse metadata encoded in the PNG filename.

    Convention: {subject}__{subject}_{contrast}__{orientation}__{slice}__{t}__{sp}.png
    """
    stem = Path(filename).stem
    parts = stem.split("__")

    contrast = "unknown"
    orientation = "unknown"

    if len(parts) >= 2:
        m = _CONTRAST_RE.match(parts[1])
        if m:
            contrast = m.group(1)

    if len(parts) >= 3:
        orientation = parts[2]

    return {"contrast": contrast, "orientation": orientation}


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

def load_model(model_name: str, device: torch.device):
    print(f"Loading model  : {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
    return model, processor


# ─── Embedding extraction ─────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(df: pd.DataFrame, model, processor,
                       device: torch.device, batch_size: int) -> np.ndarray:
    paths = df["path"].tolist()
    all_embs: list[np.ndarray] = []

    for i in tqdm(range(0, len(paths), batch_size), desc="Embeddings"):
        batch = paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs)
        # CLS token → first token of last hidden state
        cls = out.last_hidden_state[:, 0, :].cpu().float().numpy()
        all_embs.append(cls)

    return np.vstack(all_embs)


# ─── UMAP ────────────────────────────────────────────────────────────────────

def run_umap(embeddings: np.ndarray,
             n_neighbors: int, min_dist: float) -> np.ndarray:
    print("Running UMAP…")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist,
        n_components=2, random_state=42, verbose=True,
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

def linear_probing(embeddings: np.ndarray, df: pd.DataFrame,
                   columns: list[str], n_splits: int = 5) -> pd.DataFrame:
    scaler = StandardScaler()
    X = scaler.fit_transform(embeddings)
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

        clf = LogisticRegression(
            max_iter=2000, C=1.0, solver="lbfgs",
            class_weight="balanced", multi_class="auto", random_state=42,
        )

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            clf.fit(X[train_idx], y[train_idx])
            pred = clf.predict(X[val_idx])
            scores.append(balanced_accuracy_score(y[val_idx], pred))

        mean, std = float(np.mean(scores)), float(np.std(scores))
        chance = 1.0 / len(classes)
        print(f"  {col:30s}  bal_acc={mean:.3f} ± {std:.3f}  "
              f"(chance={chance:.3f}, n_classes={len(classes)})")
        rows.append({
            "attribute":         col,
            "n_classes":         len(classes),
            "classes":           ", ".join(classes),
            "chance":            round(chance, 4),
            "bal_acc_mean":      round(mean, 4),
            "bal_acc_std":       round(std, 4),
            "delta_over_chance": round(mean - chance, 4),
        })

    return pd.DataFrame(rows)


def plot_linear_probing(results: pd.DataFrame, output_dir: Path) -> None:
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
    ax.set_title("Linear probing separability by attribute")
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.tight_layout()
    out_path = output_dir / "linear_probing.png"
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
    parser.add_argument("--umap_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--no_umap",            action="store_true")
    parser.add_argument("--no_linear_probing",  action="store_true")
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

    splits = ["train", "val"] if args.split == "both" else [args.split]

    # ── Collect files ──
    print("Scanning files…")
    df = collect_files(Path(args.data_dir), splits, args.max_per_dataset)
    if df.empty:
        print("No PNG files found. Check --data_dir.")
        return
    print(f"Found {len(df):,} images across {df['dataset'].nunique()} datasets")
    print(df[["split", "dataset", "contrast", "orientation",
              "body_part", "pathology"]].describe(include="all").to_string())

    # ── Model + embeddings ──
    model, processor = load_model(args.model, device)
    embeddings = extract_embeddings(df, model, processor, device, args.batch_size)
    print(f"Embedding shape: {embeddings.shape}")

    np.save(output_dir / "embeddings.npy", embeddings)
    df.to_csv(output_dir / "metadata.csv", index=False)
    print("Saved embeddings.npy + metadata.csv")

    probe_cols = ["contrast", "orientation", "body_part", "pathology", "split"]
    umap_cols  = ["contrast", "orientation", "body_part", "pathology",
                  "dataset", "split"]

    # ── UMAP ──
    if not args.no_umap:
        coords = run_umap(embeddings, args.umap_neighbors, args.umap_min_dist)
        np.save(output_dir / "umap_coords.npy", coords)
        print("Plotting UMAPs…")
        plot_umap(coords, df, output_dir, umap_cols)

    # ── Linear probing ──
    if not args.no_linear_probing:
        print("Running linear probing…")
        results = linear_probing(embeddings, df, probe_cols)
        results.to_csv(output_dir / "linear_probing_results.csv", index=False)
        plot_linear_probing(results, output_dir)
        print("\n" + results.to_string(index=False))

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
