"""
Évaluation sur le test set avec métriques complètes.

Usage:
    python -m segmentation_hf.evaluate_test \
        --checkpoint outputs_seg/fold_0/best.pt \
        --model_dir  /path/to/curia \
        --test_npz_dir segmentation_hf/data/test_npz \
        --output_dir   outputs_seg/fold_0/test_eval \
        [--image_size 224] [--amp] [--batch_size 64]
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from tqdm import tqdm

from .dataset import NpzSegmentationDataset
from .model import FrozenBackboneWithSegHead


# ── Metric helpers ────────────────────────────────────────────────────────────

def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th percentile Hausdorff distance on 2D binary masks. NaN if one mask is empty."""
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return float("nan")
    pred_b = pred & ~binary_erosion(pred)
    gt_b   = gt   & ~binary_erosion(gt)
    pred_pts = np.argwhere(pred_b)
    gt_pts   = np.argwhere(gt_b)
    d_p2g = cKDTree(gt_pts).query(pred_pts)[0]
    d_g2p = cKDTree(pred_pts).query(gt_pts)[0]
    return float(np.percentile(np.concatenate([d_p2g, d_g2p]), 95))


def binary_metrics(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> dict:
    """Compute Dice, IoU, Precision, Recall, Specificity from binary arrays."""
    tp = float((pred & gt).sum())
    fp = float((pred & ~gt).sum())
    fn = float((~pred & gt).sum())
    tn = float((~pred & ~gt).sum())

    dice        = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou         = (tp + eps) / (tp + fp + fn + eps)
    precision   = (tp + eps) / (tp + fp + eps)
    recall      = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)

    return {
        "dice": dice, "iou": iou,
        "precision": precision, "recall": recall, "specificity": specificity,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "has_fg": int(gt.sum() > 0),
        "hd95": hd95(pred, gt),
    }


def aggregate_metrics(rows: list[dict], keys=("dice", "iou", "precision", "recall", "specificity", "hd95")) -> dict:
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows if not np.isnan(r.get(k, float("nan")))], dtype=np.float64)
        if len(vals) == 0:
            out[k] = {}
            continue
        out[k] = {
            "mean":   float(vals.mean()),
            "std":    float(vals.std()),
            "median": float(np.median(vals)),
            "p25":    float(np.percentile(vals, 25)),
            "p75":    float(np.percentile(vals, 75)),
            "min":    float(vals.min()),
            "max":    float(vals.max()),
        }
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--model_dir",    default=None)
    p.add_argument("--in_channels",  type=int, default=None,
                   help="Token dim for non-HF backbones (e.g. MRICore=256). Skips backbone loading.")
    p.add_argument("--test_npz_dir", required=True)
    p.add_argument("--output_dir",   required=True)
    p.add_argument("--image_size",   type=int, default=224)
    p.add_argument("--batch_size",   type=int, default=64)
    p.add_argument("--amp",          action="store_true")
    p.add_argument("--patch_token_key", default="patch_tokens")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ───────────────────────────────────────────────────────────
    # Auto-detect best_params.json (checkpoint is at output_dir/final/best.pt)
    ckpt_path   = Path(args.checkpoint)
    params_json = ckpt_path.parent.parent / "best_params.json"
    arch_kwargs = {}
    if params_json.exists():
        bp = json.loads(params_json.read_text())
        for k in ("seg_head_channels", "seg_head_depth", "seg_head_dropout", "seg_head_norm", "seg_head_nonlin"):
            if k in bp:
                arch_kwargs[k] = bp[k]
        print(f"Architecture depuis {params_json.name} : {arch_kwargs}")

    model = FrozenBackboneWithSegHead(args.model_dir, **arch_kwargs,
                                       in_channels=args.in_channels).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    # strip torch.compile() _orig_mod prefix if present
    if any(k.startswith("seg_head._orig_mod.") for k in state):
        state = {k.replace("seg_head._orig_mod.", "seg_head."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print(f"Modèle chargé depuis : {args.checkpoint}")

    # ── Load dataset ─────────────────────────────────────────────────────────
    ds = NpzSegmentationDataset(
        data_dir=args.test_npz_dir,
        image_size=args.image_size,
        token_key=args.patch_token_key,
        augment=False,
        preload=False,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=(device.type == "cuda"), drop_last=False,
    )
    print(f"Test set : {len(ds)} images dans {args.test_npz_dir}")

    target_hw = (args.image_size, args.image_size)
    amp_ctx   = torch.amp.autocast("cuda", enabled=args.amp) if device.type == "cuda" else torch.no_grad()

    rows: list[dict] = []

    with torch.no_grad():
        sample_idx = 0
        for batch_tokens, batch_masks in tqdm(loader, desc="test eval"):
            batch_tokens = batch_tokens.to(device, non_blocking=True)
            batch_masks  = batch_masks.to(device, non_blocking=True)

            with amp_ctx if args.amp else torch.no_grad():
                if args.amp and device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        logits = model.forward_from_tokens(batch_tokens, target_hw)
                else:
                    logits = model.forward_from_tokens(batch_tokens, target_hw)

            probs = torch.sigmoid(logits)

            for i in range(probs.shape[0]):
                npz_path = ds.npz_paths[sample_idx]
                name     = npz_path.stem
                subject  = name.split("__")[0]

                pred_np = (probs[i, 0].cpu().numpy() > 0.5)
                gt_np   = (batch_masks[i, 0].cpu().numpy() > 0.5)

                m = binary_metrics(pred_np, gt_np)
                m["name"]    = name
                m["subject"] = subject
                rows.append(m)
                sample_idx += 1

    # ── Per-image CSV ─────────────────────────────────────────────────────────
    csv_path = out_dir / "test_metrics_per_image.csv"
    fieldnames = ["name", "subject", "has_fg", "dice", "iou", "precision", "recall", "specificity", "hd95", "tp", "fp", "fn", "tn"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fieldnames})
    print(f"CSV écrit : {csv_path}")

    # ── Aggregate summary ─────────────────────────────────────────────────────
    rows_fg  = [r for r in rows if r["has_fg"]]
    rows_all = rows

    subjects = sorted(set(r["subject"] for r in rows))
    per_subject = {}
    for subj in subjects:
        subj_rows = [r for r in rows if r["subject"] == subj]
        subj_fg   = [r for r in subj_rows if r["has_fg"]]
        per_subject[subj] = {
            "n_slices":    len(subj_rows),
            "n_fg_slices": len(subj_fg),
            "all_slices":  aggregate_metrics(subj_rows),
            "fg_slices":   aggregate_metrics(subj_fg) if subj_fg else {},
        }

    summary = {
        "n_images":    len(rows),
        "n_fg_images": len(rows_fg),
        "all_slices":  aggregate_metrics(rows_all),
        "fg_slices":   aggregate_metrics(rows_fg) if rows_fg else {},
        "per_subject": per_subject,
    }

    json_path = out_dir / "test_metrics_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary JSON : {json_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n── Résultats (toutes les slices) ──")
    for k in ("dice", "iou", "precision", "recall", "specificity", "hd95"):
        s = summary["all_slices"].get(k, {})
        if s:
            print(f"  {k:12s} : {s['mean']:.4f} ± {s['std']:.4f}  [median={s['median']:.4f}, P25={s['p25']:.4f}, P75={s['p75']:.4f}]")

    if rows_fg:
        print(f"\n── Résultats (slices avec foreground, n={len(rows_fg)}) ──")
        for k in ("dice", "iou", "precision", "recall", "specificity", "hd95"):
            s = summary["fg_slices"].get(k, {})
            if s:
                print(f"  {k:12s} : {s['mean']:.4f} ± {s['std']:.4f}  [median={s['median']:.4f}, P25={s['p25']:.4f}, P75={s['p75']:.4f}]")


if __name__ == "__main__":
    main()
