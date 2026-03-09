"""
Parse bootstrap_results.csv and plot the effect of dilation radius
on classification metrics (with 95% CI error bars).

Usage:
    python -m classification_hf.plot_dilation_study \
        --log_dir classification_hf/logs \
        --tasks nfn scs ss \
        --radii 0 2 4 6 8 12 16 24

Can also be run standalone after a dilation study to regenerate the figure.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TASK_LABELS = {
    "nfn": "Neural Foraminal Narrowing",
    "scs": "Spinal Canal Stenosis",
    "ss":  "Subarticular Stenosis",
}
COLORS = ["steelblue", "crimson", "darkorange", "forestgreen"]
LINESTYLES = ["-", "--"]   # solid = first variant, dashed = second
# (csv_prefix, axis label, best=max)
METRICS = [
    ("auc_macro",    "AUC OvR macro",    "max"),
    ("auc_weighted", "AUC OvR weighted", "max"),
    ("accuracy",     "Accuracy",         "max"),
]


def load_results(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")
    return list(csv.DictReader(csv_path.open(encoding="utf-8")))


def extract_dilation_rows(rows: list[dict], base_task: str, radii: list[int]) -> dict[int, dict]:
    """Keep the most recent row per (task, dilation_radius) pair."""
    result = {}
    for r in rows:
        if r["task"] != base_task:
            continue
        try:
            dil = int(r["dilation_radius"])
        except (KeyError, ValueError):
            continue
        if dil in radii:
            result[dil] = r   # last write wins (most recent run)
    return result


def _annotate_best(ax, x_vals, y_vals, mode, color):
    arr = np.array(y_vals, dtype=float)
    if np.all(np.isnan(arr)):
        return
    idx = int(np.nanargmax(arr)) if mode == "max" else int(np.nanargmin(arr))
    bx, by = x_vals[idx], arr[idx]
    ax.plot(bx, by, "o", color=color, markersize=9, zorder=5)
    ax.annotate(
        f"{by:.4f}\n@dil={bx}",
        xy=(bx, by),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=7.5,
        color=color,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
    )


def _plot_variant(axes, rows, tasks, radii, variant_label, linestyle):
    """Plot one variant's curves onto existing axes."""
    for ax, (col_prefix, label, mode) in zip(axes, METRICS):
        for task, color in zip(tasks, COLORS):
            dil_rows = extract_dilation_rows(rows, task, radii)
            if not dil_rows:
                continue
            x = sorted(dil_rows.keys())
            y    = [float(dil_rows[d].get(f"{col_prefix}_mean",    "nan")) for d in x]
            y_lo = [float(dil_rows[d].get(f"{col_prefix}_ci_low",  "nan")) for d in x]
            y_hi = [float(dil_rows[d].get(f"{col_prefix}_ci_high", "nan")) for d in x]
            y_arr    = np.array(y,    dtype=float)
            err_lo   = y_arr - np.array(y_lo, dtype=float)
            err_hi   = np.array(y_hi, dtype=float) - y_arr
            task_label = f"{TASK_LABELS.get(task, task)} [{variant_label}]"
            ax.errorbar(x, y_arr, yerr=[err_lo, err_hi],
                        marker="s", color=color, linestyle=linestyle,
                        linewidth=2, markersize=6, capsize=4,
                        label=task_label, alpha=0.85)
            if linestyle == "-":   # annotate best only for first variant
                _annotate_best(ax, x, y, mode, color)


def plot_dilation_study(log_dir: Path, tasks: list[str], radii: list[int],
                        compare_log_dir: Path | None = None,
                        labels: list[str] | None = None,
                        out: Path | None = None) -> None:
    comparing = compare_log_dir is not None
    label_a = (labels[0] if labels else "default")
    label_b = (labels[1] if labels and len(labels) > 1 else "custom")

    rows_a = load_results(log_dir / "bootstrap_results.csv")
    rows_b = load_results(compare_log_dir / "bootstrap_results.csv") if comparing else None

    n_metrics = len(METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    title = "Dilation radius ablation — bootstrap 95% CI"
    if comparing:
        title += f"  [{label_a}  vs  {label_b}]"
    fig.suptitle(title, fontsize=13)

    _plot_variant(axes, rows_a, tasks, radii, label_a, LINESTYLES[0])
    if comparing:
        _plot_variant(axes, rows_b, tasks, radii, label_b, LINESTYLES[1])

    all_radii = sorted(set(radii))
    for ax, (_, label, _) in zip(axes, METRICS):
        ax.set_title(label)
        ax.set_xlabel("Dilation radius (pixels)")
        ax.set_xticks(all_radii)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes:
        ax.set_ylim(bottom=max(0.0, ax.get_ylim()[0] - 0.02))

    plt.tight_layout()
    out_path = out or (log_dir / "dilation_study.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Dilation study plot saved to {out_path}")

    # Print summary table
    for variant_label, rows in [(label_a, rows_a)] + ([(label_b, rows_b)] if comparing else []):
        print(f"\n{'='*80}  [{variant_label}]")
        print(f"{'Task':<10}  {'Dil':>4}  {'AUC macro':>10}  {'AUC weighted':>12}  {'Accuracy':>10}")
        print(f"{'─'*80}")
        for task in tasks:
            dil_rows = extract_dilation_rows(rows, task, radii)
            for dil in sorted(dil_rows.keys()):
                r = dil_rows[dil]
                print(f"{task:<10}  {dil:>4}  "
                      f"{float(r.get('auc_macro_mean', 0)):>10.4f}  "
                      f"{float(r.get('auc_weighted_mean', 0)):>12.4f}  "
                      f"{float(r.get('accuracy_mean', 0)):>10.4f}")
            print()
        print("="*80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir",         required=True)
    parser.add_argument("--compare_log_dir", type=Path, default=None,
                        help="Second log dir to overlay on the same plot")
    parser.add_argument("--labels",  nargs="+", default=None,
                        help="Legend labels for the two variants (default: 'default' 'custom')")
    parser.add_argument("--tasks",   nargs="+", default=["nfn", "scs", "ss"])
    parser.add_argument("--radii",   nargs="+", type=int,
                        default=[0, 2, 4, 6, 8, 12, 16, 24])
    parser.add_argument("--out",     type=Path, default=None,
                        help="Output PNG path (default: <log_dir>/dilation_study.png)")
    args = parser.parse_args()

    plot_dilation_study(
        Path(args.log_dir), args.tasks, args.radii,
        compare_log_dir=args.compare_log_dir,
        labels=args.labels,
        out=args.out,
    )


if __name__ == "__main__":
    main()
