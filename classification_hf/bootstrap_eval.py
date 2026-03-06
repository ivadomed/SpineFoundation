"""
Bootstrap evaluation — two modes:

  trained    Pool val_predictions.npz from N training run subdirectories,
             then bootstrap to estimate 95% CIs.

  pretrained Run inference with a frozen HuggingFace head (e.g. raidium/curia)
             on cached patch_tokens, then bootstrap.

Usage — trained mode:
    python -m classification_hf.bootstrap_eval \
        --task nfn \
        --runs_dir /home/ge.polymtl.ca/p123239/SpineFoundation/outputs_cls/rsna_nfn

Usage — pretrained mode:
    python -m classification_hf.bootstrap_eval \
        --mode pretrained \
        --task nfn

    python -m classification_hf.bootstrap_eval \
        --mode pretrained \
        --task scs \
        --dilation-radius 8 \
        --model-name raidium/curia
"""

import argparse
import csv
import fcntl
import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_DATA_ROOT = Path("/home/ge.polymtl.ca/p123239/data")
TASK_CONFIG = {
    "nfn": {
        "subfolder": "neural_foraminal_narrowing",
        "data_dir":  _DATA_ROOT / "patches_RSNA_raw_with_mask_nfn",
    },
    "ss": {
        "subfolder": "subarticular_stenosis",
        "data_dir":  _DATA_ROOT / "patches_RSNA_raw_with_mask_ss",
    },
    "scs": {
        "subfolder": "spinal_canal_stenosis",
        "data_dir":  _DATA_ROOT / "patches_RSNA_raw_with_mask_scs",
    },
}


# ── Bootstrap core ────────────────────────────────────────────────────────────


def _compute_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    """Compute accuracy + AUC on a single (possibly bootstrapped) sample."""
    proba = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    try:
        auc_macro    = roc_auc_score(labels, proba, multi_class="ovr", average="macro")
        auc_weighted = roc_auc_score(labels, proba, multi_class="ovr", average="weighted")
    except ValueError:
        auc_macro = auc_weighted = float("nan")
    return {"accuracy": acc, "auc_ovr_macro": auc_macro, "auc_ovr_weighted": auc_weighted}


def bootstrap(logits: np.ndarray, labels: np.ndarray, n_bootstrap: int, seed: int) -> dict:
    """
    Draw `n_bootstrap` resamples with replacement; return mean + 95% CI per metric.
    Returns dict: {metric: {"mean": float, "ci_low": float, "ci_high": float}}
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    samples: dict = {"accuracy": [], "auc_ovr_macro": [], "auc_ovr_weighted": []}
    for _ in tqdm(range(n_bootstrap), desc="Bootstrapping", unit="resample", leave=False):
        idx = rng.integers(0, n, size=n)
        m = _compute_metrics(logits[idx], labels[idx])
        for k, v in m.items():
            samples[k].append(v)
    result = {}
    for metric, vals in samples.items():
        arr = np.array(vals, dtype=float)
        result[metric] = {
            "mean":     float(np.nanmean(arr)),
            "ci_low":   float(np.nanpercentile(arr, 2.5)),
            "ci_high":  float(np.nanpercentile(arr, 97.5)),
        }
    return result


# ── Trained mode: pool predictions from training run subdirectories ───────────


def _find_prediction_files(runs_dir: Path) -> list[Path]:
    """Return all val_predictions.npz files inside runs_dir subdirectories."""
    pred_files = sorted(runs_dir.glob("**/val_predictions.npz"))
    if not pred_files:
        raise FileNotFoundError(
            f"No val_predictions.npz found under {runs_dir}\n"
            "Make sure training runs completed (trainer saves predictions automatically)."
        )
    return pred_files


def _load_and_pool(pred_files: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Load and vertically stack (logits, labels) from all prediction files."""
    all_logits, all_labels = [], []
    for path in pred_files:
        d = np.load(str(path))
        all_logits.append(d["logits"].astype(np.float32))
        all_labels.append(d["labels"].astype(np.int64))
        print(f"  loaded {path.parent.name}/val_predictions.npz  — {len(d['labels'])} samples")
    return np.concatenate(all_logits), np.concatenate(all_labels)


# ── Pretrained mode: inference with frozen HuggingFace head ──────────────────


def _make_mask_transform(mask_np: np.ndarray, target_size: int) -> torch.Tensor:
    t = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(target_size, target_size),
                      mode="bilinear", align_corners=False, antialias=True)
    t = t.squeeze(0)
    mask = t > 0.5
    if mask.sum() == 0:
        mask = t > (t.max() * 0.5)
    return mask.float().unsqueeze(0)   # (1, 1, S, S)


def _masked_avg_pool(patch_tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """patch_tokens: (B, N, D)  mask: (B, 1, S, S) → (B, D)"""
    B, N, D = patch_tokens.shape
    grid = int(N ** 0.5)
    patch_size = mask.shape[-1] // grid
    mask_pooled = F.max_pool2d(mask.float(), kernel_size=patch_size, stride=patch_size)
    mask_flat = mask_pooled.view(B, N).unsqueeze(-1)
    total = mask_flat.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return (patch_tokens * mask_flat).sum(dim=1) / total.squeeze(1)


def _collect_paths_and_labels(data_dir: Path) -> tuple[list[Path], np.ndarray]:
    """Collect (path, label) pairs from class-named subdirectories."""
    paths: list[Path] = []
    labels: list[int] = []
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        try:
            cls = int(class_dir.name)
        except ValueError:
            continue
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() == ".npz" and not f.name.startswith("tmp"):
                paths.append(f)
                labels.append(cls)
    return paths, np.array(labels)


def _run_pretrained_inference(args, paths: list[Path], crop_size: int,
                              classifier) -> np.ndarray:
    """Returns logits (N, C) using cached patch_tokens + masked avg pooling."""
    _SENTINEL = object()
    _q: queue.Queue = queue.Queue(maxsize=2)
    _pool = ThreadPoolExecutor(max_workers=16)

    def _load_npz(p: Path) -> tuple[np.ndarray, np.ndarray]:
        d = np.load(p)
        return d["patch_tokens"], d["mask"]

    def _apply_dilation(m: torch.Tensor) -> torch.Tensor:
        if args.dilation_radius > 0:
            k = 2 * args.dilation_radius + 1
            return F.max_pool2d(m, kernel_size=k, stride=1, padding=args.dilation_radius)
        return m

    def _feeder(paths_: list, batch_size_: int, q: queue.Queue) -> None:
        for i in range(0, len(paths_), batch_size_):
            bp = paths_[i:i + batch_size_]
            loaded = list(_pool.map(_load_npz, bp))
            tokens_list, masks_list = zip(*loaded)
            tokens_t = torch.from_numpy(np.stack(tokens_list))
            mask_t = torch.cat(
                [_make_mask_transform(m, crop_size) for m in masks_list], dim=0
            )
            q.put((tokens_t, mask_t))
        q.put(_SENTINEL)

    threading.Thread(
        target=_feeder, args=(paths, args.batch_size, _q), daemon=True
    ).start()

    all_logits: list[np.ndarray] = []
    with tqdm(total=len(paths), desc=f"Inference ({args.task})", unit="ex") as pbar:
        while True:
            item = _q.get()
            if item is _SENTINEL:
                break
            tokens_cpu, mask_cpu = item
            with torch.no_grad():
                tokens = tokens_cpu.to(DEVICE)
                mask   = _apply_dilation(mask_cpu.to(DEVICE))
                pooled = _masked_avg_pool(tokens, mask)
                logits = classifier(pooled)
            all_logits.append(logits.cpu().numpy())
            pbar.update(tokens_cpu.shape[0])

    _pool.shutdown(wait=False)
    return np.concatenate(all_logits, axis=0)


# ── Shared output helpers ─────────────────────────────────────────────────────


def _print_table_row(task: str, result: dict, extra: str = "") -> str:
    cols = []
    for metric in ["auc_ovr_macro", "auc_ovr_weighted", "accuracy"]:
        r = result[metric]
        cols.append(f"{r['mean']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")
    label = f"{task}{extra}"
    return f"  {label:<12}  {cols[0]:<28}  {cols[1]:<28}  {cols[2]}"


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Bootstrap evaluation")
    parser.add_argument("--mode",    choices=["trained", "pretrained"], default="trained")
    parser.add_argument("--task",    required=True, choices=["nfn", "ss", "scs"])
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--log-dir", type=Path, default=None,
                        help="Directory to write results (default: classification_hf/logs)")

    # Trained mode args
    parser.add_argument("--runs-dir", type=Path, default=None,
                        help="[trained] Base output dir containing run subdirs with val_predictions.npz")

    # Pretrained mode args
    parser.add_argument("--model-name",     type=str, default="raidium/curia",
                        help="[pretrained] HuggingFace model id")
    parser.add_argument("--subfolder",      type=str, default=None,
                        help="[pretrained] Task-specific subfolder (default: from TASK_CONFIG)")
    parser.add_argument("--data-dir",       type=Path, default=None,
                        help="[pretrained] Directory with class-named subdirs of NPZ patches")
    parser.add_argument("--dilation-radius", type=int, default=0,
                        help="[pretrained] Mask dilation radius in pixels")
    parser.add_argument("--batch-size",     type=int, default=128,
                        help="[pretrained] Inference batch size")

    args = parser.parse_args()

    # Apply TASK_CONFIG defaults for pretrained mode
    cfg = TASK_CONFIG[args.task]
    if args.subfolder is None:
        args.subfolder = cfg["subfolder"]
    if args.data_dir is None:
        args.data_dir = cfg["data_dir"]

    # Default log dir
    _here = Path(__file__).parent
    log_dir = args.log_dir or (_here / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Banner ────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"Bootstrap evaluation — mode: {args.mode}   task: {args.task}")
    if args.mode == "pretrained":
        print(f"Model      : {args.model_name}/{args.subfolder}")
        print(f"Data dir   : {args.data_dir}")
        print(f"Device     : {DEVICE}")
        if args.dilation_radius > 0:
            print(f"Dilation   : {args.dilation_radius}px")
    else:
        print(f"Runs dir   : {args.runs_dir}")
    print(f"Bootstraps : {args.n_bootstrap}   seed={args.seed}")
    print(f"{'='*65}\n")

    # ── Data loading / inference ───────────────────────────────────────────────
    if args.mode == "trained":
        if args.runs_dir is None:
            parser.error("--runs-dir is required for --mode trained")
        pred_files = _find_prediction_files(args.runs_dir)
        n_runs = len(pred_files)
        print(f"Found {n_runs} run(s) with predictions:")
        logits, labels = _load_and_pool(pred_files)
    else:
        # Pretrained mode
        paths, labels = _collect_paths_and_labels(args.data_dir)
        if not paths:
            raise RuntimeError(f"No NPZ files found in {args.data_dir}")
        print(f"Total: {len(paths)} samples  (class distribution: {np.bincount(labels).tolist()})")

        # Verify cache
        probe = np.load(paths[0])
        if "patch_tokens" not in probe.files:
            raise RuntimeError(
                f"No 'patch_tokens' key in {paths[0]}.\n"
                "Run cache_patch_tokens.py first to cache patch tokens."
            )

        print("Loading model...")
        processor = AutoImageProcessor.from_pretrained(args.model_name, trust_remote_code=True)
        model = AutoModelForImageClassification.from_pretrained(
            args.model_name, subfolder=args.subfolder, trust_remote_code=True
        )
        model.eval().to(DEVICE)
        classifier = model.classifier.to(DEVICE)

        _cs = processor.crop_size
        crop_size = _cs["height"] if isinstance(_cs, dict) else int(_cs)

        logits = _run_pretrained_inference(args, paths, crop_size, classifier)

    print(f"\nPooled dataset: {len(labels)} samples  "
          f"(class distribution: {np.bincount(labels).tolist()})")

    # ── Point estimates ───────────────────────────────────────────────────────
    point = _compute_metrics(logits, labels)
    print(f"\nPoint estimates (full dataset, no bootstrap):")
    for k in ["auc_ovr_macro", "auc_ovr_weighted", "accuracy"]:
        print(f"  {k:<22}: {point[k]:.4f}")

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    print(f"\nRunning {args.n_bootstrap} bootstrap resamples ...")
    result = bootstrap(logits, labels, args.n_bootstrap, args.seed)

    # ── Pretty print ──────────────────────────────────────────────────────────
    dil_str = f" dil={args.dilation_radius}px" if args.mode == "pretrained" and args.dilation_radius > 0 else ""
    header = (
        f"\n{'─'*90}\n"
        f"  {'Task':<12}  {'AUC macro  mean [95% CI]':<28}  "
        f"{'AUC weighted  mean [95% CI]':<28}  Accuracy  mean [95% CI]\n"
        f"{'─'*90}"
    )
    print(header)
    print(_print_table_row(args.task, result, extra=dil_str))
    print(f"{'─'*90}\n")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    dil_tag = f"_dil{args.dilation_radius}" if args.mode == "pretrained" and args.dilation_radius > 0 else ""
    summary: dict = {
        "timestamp":       timestamp,
        "mode":            args.mode,
        "task":            args.task,
        "n_samples":       int(len(labels)),
        "n_bootstrap":     args.n_bootstrap,
        "seed":            args.seed,
        "point_estimates": {k: round(v, 6) for k, v in point.items()},
        "bootstrap": {
            m: {k2: round(v2, 6) for k2, v2 in vals.items()}
            for m, vals in result.items()
        },
    }
    if args.mode == "trained":
        summary["n_runs"] = n_runs
    else:
        summary["model"]           = args.model_name
        summary["subfolder"]       = args.subfolder
        summary["dilation_radius"] = args.dilation_radius

    json_name = (
        f"bootstrap_pretrained_{args.task}{dil_tag}.json"
        if args.mode == "pretrained"
        else f"bootstrap_{args.task}.json"
    )
    json_path = log_dir / json_name
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"Full summary saved to {json_path}")

    # ── Append to CSV ─────────────────────────────────────────────────────────
    if args.mode == "trained":
        csv_path = log_dir / "bootstrap_results.csv"
        fieldnames = [
            "timestamp", "task", "n_runs", "n_samples", "n_bootstrap",
            "accuracy_mean", "accuracy_ci_low", "accuracy_ci_high",
            "auc_macro_mean", "auc_macro_ci_low", "auc_macro_ci_high",
            "auc_weighted_mean", "auc_weighted_ci_low", "auc_weighted_ci_high",
        ]
        row = {
            "timestamp":   timestamp,
            "task":        args.task,
            "n_runs":      n_runs,
            "n_samples":   len(labels),
            "n_bootstrap": args.n_bootstrap,
        }
    else:
        csv_path = log_dir / "bootstrap_pretrained_results.csv"
        fieldnames = [
            "timestamp", "task", "dilation_radius", "model", "n_samples", "n_bootstrap",
            "accuracy_mean", "accuracy_ci_low", "accuracy_ci_high",
            "auc_macro_mean", "auc_macro_ci_low", "auc_macro_ci_high",
            "auc_weighted_mean", "auc_weighted_ci_low", "auc_weighted_ci_high",
        ]
        row = {
            "timestamp":       timestamp,
            "task":            args.task,
            "dilation_radius": args.dilation_radius,
            "model":           f"{args.model_name}/{args.subfolder}",
            "n_samples":       len(labels),
            "n_bootstrap":     args.n_bootstrap,
        }

    for metric, col_prefix in [("accuracy", "accuracy"), ("auc_ovr_macro", "auc_macro"),
                                ("auc_ovr_weighted", "auc_weighted")]:
        row[f"{col_prefix}_mean"]    = round(result[metric]["mean"],    6)
        row[f"{col_prefix}_ci_low"]  = round(result[metric]["ci_low"],  6)
        row[f"{col_prefix}_ci_high"] = round(result[metric]["ci_high"], 6)

    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)   # NFS-safe exclusive lock
        try:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            if fh.tell() == 0:            # file was empty when we locked it
                w.writeheader()
            w.writerow(row)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
    print(f"Row appended to {csv_path}")

    # ── Final 3-task summary table ────────────────────────────────────────────
    if csv_path.exists():
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
        latest: dict = {}
        for r in rows:
            if args.mode == "pretrained":
                key = (r["task"], r["dilation_radius"])
            else:
                key = r["task"]
            latest[key] = r

        all_tasks = ["nfn", "scs", "ss"]
        if args.mode == "pretrained":
            dil_key = str(args.dilation_radius)
            have_all = all((t, dil_key) in latest for t in all_tasks)
        else:
            have_all = all(t in latest for t in all_tasks)

        if have_all:
            print(f"\n{'='*90}")
            label = "pretrained head" if args.mode == "pretrained" else "trained runs"
            print(f"FINAL SUMMARY ({label} — all 3 tasks)")
            print(f"{'─'*90}")
            print(f"  {'Task':<12}  {'AUC macro  mean [95% CI]':<28}  "
                  f"{'AUC weighted  mean [95% CI]':<28}  Accuracy  mean [95% CI]")
            print(f"{'─'*90}")
            for t in all_tasks:
                key = (t, dil_key) if args.mode == "pretrained" else t
                r = latest[key]
                mac_s = f"{r['auc_macro_mean']}  [{r['auc_macro_ci_low']}, {r['auc_macro_ci_high']}]"
                wgt_s = f"{r['auc_weighted_mean']}  [{r['auc_weighted_ci_low']}, {r['auc_weighted_ci_high']}]"
                acc_s = f"{r['accuracy_mean']}  [{r['accuracy_ci_low']}, {r['accuracy_ci_high']}]"
                print(f"  {t:<12}  {mac_s:<28}  {wgt_s:<28}  {acc_s}")
            print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
