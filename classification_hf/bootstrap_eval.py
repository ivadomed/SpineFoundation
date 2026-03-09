"""
Bootstrap evaluation — two modes:

  trained    Pool val_predictions.npz from N training run subdirectories,
             then bootstrap to estimate 95% CIs.

  pretrained Load pre-cached pooled features (.pt produced by
             cache_pooled_features.py), run the frozen HuggingFace classifier
             head, then bootstrap.

Usage — trained mode:
    python -m classification_hf.bootstrap_eval \
        --task nfn \
        --dilation-radius 8 \
        --runs-dir /home/ge.polymtl.ca/p123239/SpineFoundation/outputs_cls/rsna_nfn_dil8

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
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForImageClassification


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_DATA_ROOT = Path("/home/ge.polymtl.ca/p123239/data")
TASK_CONFIG = {
    "nfn": {
        "subfolder": "neural_foraminal_narrowing",
        "data_dir":  _DATA_ROOT / "RSNA_patches_nfn",
    },
    "ss": {
        "subfolder": "subarticular_stenosis",
        "data_dir":  _DATA_ROOT / "RSNA_patches_ss",
    },
    "scs": {
        "subfolder": "spinal_canal_stenosis",
        "data_dir":  _DATA_ROOT / "RSNA_patches_scs",
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


# ── Pretrained mode: load cached pooled features + frozen classifier ──────────


def _load_pooled_features(data_dir: Path, dilation_radius: int) -> tuple[torch.Tensor, np.ndarray]:
    """Load pooled features and labels from cache_pooled_features output."""
    cache_path = data_dir / f"pooled_features_dil{dilation_radius}.pt"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Pooled features cache not found: {cache_path}\n"
            "Run cache_pooled_features.py first (called automatically by run_dilation_study.sh)."
        )
    print(f"Loading pooled features from {cache_path}")
    cached = torch.load(cache_path, map_location="cpu", weights_only=True)
    features = cached["features"]          # (N, D)
    labels   = cached["labels"].numpy()    # (N,)
    return features, labels


# ── Shared output helpers ─────────────────────────────────────────────────────


def _print_table_row(task: str, result: dict, extra: str = "") -> str:
    cols = []
    for metric in ["auc_ovr_macro", "auc_ovr_weighted", "accuracy"]:
        r = result[metric]
        cols.append(f"{r['mean']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")
    label = f"{task}{extra}"
    return f"  {label:<12}  {cols[0]:<28}  {cols[1]:<28}  {cols[2]}"


# ── CSV helpers ───────────────────────────────────────────────────────────────


def _rebuild_csv(csv_path: Path, fieldnames: list, new_row: dict, log_dir: Path, mode: str) -> None:
    """Rebuild the CSV from all JSON files + new_row to avoid header/schema drift."""
    json_glob = "bootstrap_pretrained_*.json" if mode == "pretrained" else "bootstrap_[ns]*.json"
    rows_by_key: dict = {}

    for jf in sorted(log_dir.glob(json_glob)):
        try:
            d = json.loads(jf.read_text())
        except Exception:
            continue
        if d.get("mode") != mode:
            continue
        row = {
            "timestamp":       d["timestamp"],
            "task":            d["task"],
            "dilation_radius": d["dilation_radius"],
            "n_samples":       d["n_samples"],
            "n_bootstrap":     d["n_bootstrap"],
        }
        if mode == "trained":
            row["n_runs"] = d.get("n_runs", "")
        for metric, col in [("accuracy", "accuracy"), ("auc_ovr_macro", "auc_macro"),
                             ("auc_ovr_weighted", "auc_weighted")]:
            b = d["bootstrap"][metric]
            row[f"{col}_mean"]    = round(b["mean"],    6)
            row[f"{col}_ci_low"]  = round(b["ci_low"],  6)
            row[f"{col}_ci_high"] = round(b["ci_high"], 6)
        rows_by_key[(d["task"], d["dilation_radius"])] = row

    # New row takes priority
    rows_by_key[(new_row["task"], new_row["dilation_radius"])] = new_row

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows_by_key.values())
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Bootstrap evaluation")
    parser.add_argument("--mode",    choices=["trained", "pretrained"], default="trained")
    parser.add_argument("--task",    required=True, choices=["nfn", "ss", "scs"])
    parser.add_argument("--dilation-radius", type=int, default=0,
                        help="Dilation radius for this run (used in CSV output)")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--log-dir", type=Path, default=None,
                        help="Directory to write results (default: classification_hf/logs)")

    # Trained mode args
    parser.add_argument("--runs-dir", type=Path, default=None,
                        help="[trained] Base output dir containing run subdirs with val_predictions.npz")

    # Pretrained mode args
    parser.add_argument("--model-name",  type=str, default="raidium/curia",
                        help="[pretrained] HuggingFace model id")
    parser.add_argument("--subfolder",   type=str, default=None,
                        help="[pretrained] Task-specific subfolder (default: from TASK_CONFIG)")
    parser.add_argument("--data-dir",    type=Path, default=None,
                        help="[pretrained] Directory containing pooled_features_dil{N}.pt")

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
    else:
        print(f"Runs dir   : {args.runs_dir}")
    if args.dilation_radius > 0:
        print(f"Dilation   : {args.dilation_radius}px")
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
        # Pretrained mode — load pre-cached pooled features
        features, labels = _load_pooled_features(args.data_dir, args.dilation_radius)
        print(f"Total: {len(labels)} samples  (class distribution: {np.bincount(labels).tolist()})")

        print("Loading classifier...")
        model = AutoModelForImageClassification.from_pretrained(
            args.model_name, subfolder=args.subfolder, trust_remote_code=True
        )
        model.eval()
        classifier = model.classifier.to(DEVICE)

        with torch.no_grad():
            logits = classifier(features.to(DEVICE)).cpu().numpy()

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
    dil_str = f" dil={args.dilation_radius}px" if args.dilation_radius > 0 else ""
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
    dil_tag = f"_dil{args.dilation_radius}" if args.dilation_radius > 0 else ""
    summary: dict = {
        "timestamp":       timestamp,
        "mode":            args.mode,
        "task":            args.task,
        "dilation_radius": args.dilation_radius,
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
        summary["model"]     = args.model_name
        summary["subfolder"] = args.subfolder

    json_name = (
        f"bootstrap_pretrained_{args.task}{dil_tag}.json"
        if args.mode == "pretrained"
        else f"bootstrap_{args.task}{dil_tag}.json"
    )
    json_path = log_dir / json_name
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"Full summary saved to {json_path}")

    # ── Append to CSV ─────────────────────────────────────────────────────────
    csv_path = log_dir / (
        "bootstrap_pretrained_results.csv"
        if args.mode == "pretrained"
        else "bootstrap_results.csv"
    )
    fieldnames = [
        "timestamp", "task", "dilation_radius", "n_samples", "n_bootstrap",
        "accuracy_mean", "accuracy_ci_low", "accuracy_ci_high",
        "auc_macro_mean", "auc_macro_ci_low", "auc_macro_ci_high",
        "auc_weighted_mean", "auc_weighted_ci_low", "auc_weighted_ci_high",
    ]
    row = {
        "timestamp":       timestamp,
        "task":            args.task,
        "dilation_radius": args.dilation_radius,
        "n_samples":       len(labels),
        "n_bootstrap":     args.n_bootstrap,
    }
    if args.mode == "trained":
        fieldnames.insert(4, "n_runs")
        row["n_runs"] = n_runs

    for metric, col_prefix in [("accuracy", "accuracy"), ("auc_ovr_macro", "auc_macro"),
                                ("auc_ovr_weighted", "auc_weighted")]:
        row[f"{col_prefix}_mean"]    = round(result[metric]["mean"],    6)
        row[f"{col_prefix}_ci_low"]  = round(result[metric]["ci_low"],  6)
        row[f"{col_prefix}_ci_high"] = round(result[metric]["ci_high"], 6)

    # Rebuild CSV from all JSON files to avoid header/schema mismatch on append
    _rebuild_csv(csv_path, fieldnames, row, log_dir, args.mode)
    print(f"Row appended to {csv_path}")

    # ── Final 3-task summary table ────────────────────────────────────────────
    if csv_path.exists():
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
        dil_key = str(args.dilation_radius)
        latest: dict = {}
        for r in rows:
            if r.get("dilation_radius") == dil_key:
                latest[r["task"]] = r

        all_tasks = ["nfn", "scs", "ss"]
        if all(t in latest for t in all_tasks):
            print(f"\n{'='*90}")
            label = "pretrained head" if args.mode == "pretrained" else "trained runs"
            print(f"FINAL SUMMARY ({label} — all 3 tasks  dil={args.dilation_radius}px)")
            print(f"{'─'*90}")
            print(f"  {'Task':<12}  {'AUC macro  mean [95% CI]':<28}  "
                  f"{'AUC weighted  mean [95% CI]':<28}  Accuracy  mean [95% CI]")
            print(f"{'─'*90}")
            for t in all_tasks:
                r = latest[t]
                mac_s = f"{r['auc_macro_mean']}  [{r['auc_macro_ci_low']}, {r['auc_macro_ci_high']}]"
                wgt_s = f"{r['auc_weighted_mean']}  [{r['auc_weighted_ci_low']}, {r['auc_weighted_ci_high']}]"
                acc_s = f"{r['accuracy_mean']}  [{r['accuracy_ci_low']}, {r['accuracy_ci_high']}]"
                print(f"  {t:<12}  {mac_s:<28}  {wgt_s:<28}  {acc_s}")
            print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
