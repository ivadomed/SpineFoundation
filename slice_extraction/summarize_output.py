#!/usr/bin/env python3
"""
summarize_output.py

Scans an out_raw_numero-style directory and generates 3 separate PNGs:
  1. summary_datasets.png  — Dataset table (slices, axial/sagittal breakdown)
  2. summary_orientation.png — Orientation pie chart
  3. summary_contrasts.png   — Contrast bar chart

Filename convention expected:
  {split}/{class}/{dataset}__sub-{...}__...__{contrast}__{orientation}__s{N}__sp{S}x{S}__{idx}.png

Usage:
  python summarize_output.py --input-dir out_raw_numero --output-dir .
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# ── Regex patterns ─────────────────────────────────────────────────────────────
RE_DATASET   = re.compile(r"^([^_]+(?:-[^_]+)*)(?=__sub)")
RE_ORIENT    = re.compile(r"__(axial|sagittal)__")
RE_CONTRAST  = re.compile(r"_(T2starw|T2star|T1w|T2w|MTS|FLAIR|PSIR|MP2RAGE|STIR|PDw|T1rho|MTon|MToff|bold)__")

# Merge aliases
CONTRAST_ALIASES = {
    "T2starw": "T2star(w)",
    "T2star":  "T2star(w)",
    "MTon":    "MTS",
    "MToff":   "MTS",
}

# ── Parse filenames ────────────────────────────────────────────────────────────
def parse_files(root: Path):
    """
    Returns:
      datasets  : dict[dataset] -> {"total": int, "axial": int, "sagittal": int}
      contrasts : dict[contrast] -> int
      splits    : dict[split]   -> int
      orientations: {"axial": int, "sagittal": int}
    """
    datasets     = defaultdict(lambda: {"total": 0, "axial": 0, "sagittal": 0})
    contrasts    = defaultdict(int)
    splits       = defaultdict(int)
    orientations = defaultdict(int)

    # Current layout: <root>/image/<split>/<dataset>/<file>.png
    # Fallback for flat layout: <root>/<split>/<file>.png
    image_root = root / "image"
    scan_root  = image_root if image_root.exists() else root

    for png in scan_root.rglob("*.png"):
        name = png.name

        try:
            parts = png.relative_to(scan_root).parts
        except Exception:
            parts = ()

        # split = first component after scan_root (train / val)
        split = parts[0] if parts else "unknown"
        splits[split] += 1

        # dataset = subdirectory between split and filename when present
        if len(parts) >= 3:
            dataset = parts[1]   # image/train/<dataset>/file.png
        else:
            m = RE_DATASET.search(name)
            dataset = m.group(1) if m else "unknown"

        # orientation
        m = RE_ORIENT.search(name)
        orient = m.group(1) if m else "unknown"

        # contrast
        m = RE_CONTRAST.search(name)
        contrast = m.group(1) if m else "unknown"
        contrast = CONTRAST_ALIASES.get(contrast, contrast)

        datasets[dataset]["total"]  += 1
        datasets[dataset][orient]   += 1
        orientations[orient]        += 1
        contrasts[contrast]         += 1

    return datasets, contrasts, splits, orientations


# ── Drawing helpers ────────────────────────────────────────────────────────────
BG      = "white"
CELL_D  = "#f0f4f8"
CELL_L  = "white"
HDR_BG  = "#2d3a8c"
TXT     = "#1a1a2e"
TXT_HDR = "white"
AX_COL  = "#1565c0"
SAG_COL = "#e65100"
TOT_BG  = "#dce3f5"
ACCENT  = "#2d3a8c"

fmt = lambda n: f"{n:,}"


def draw_table(ax, headers, rows, col_widths, alignments=None):
    total_w = sum(col_widths)
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, len(rows) + 1)
    ax.axis("off")
    if alignments is None:
        alignments = ["left"] + ["right"] * (len(headers) - 1)

    # header
    x = 0
    for h, w in zip(headers, col_widths):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x + 0.03, len(rows) + 0.08), w - 0.06, 0.84,
            boxstyle="round,pad=0.03", facecolor=HDR_BG, edgecolor="none"))
        ax.text(x + w / 2, len(rows) + 0.50, h,
                ha="center", va="center", fontsize=9.5,
                fontweight="bold", color=TXT_HDR)
        x += w

    # rows
    for r, row in enumerate(rows):
        is_total = (r == len(rows) - 1)
        y  = len(rows) - 1 - r
        bg = TOT_BG if is_total else (CELL_D if r % 2 == 0 else CELL_L)
        x  = 0
        for i, (val, w, align) in enumerate(zip(row, col_widths, alignments)):
            ax.add_patch(mpatches.FancyBboxPatch(
                (x + 0.03, y + 0.08), w - 0.06, 0.84,
                boxstyle="round,pad=0.03", facecolor=bg, edgecolor="none"))
            xpos = x + 0.18 if align == "left" else x + w - 0.18
            fw    = "bold" if is_total else "normal"
            color = TXT
            if i == 2 and not is_total: color = AX_COL
            if i == 3 and not is_total: color = SAG_COL
            if is_total: color = ACCENT
            ax.text(xpos, y + 0.50, val,
                    ha=align, va="center", fontsize=8.5,
                    color=color, fontweight=fw)
            x += w


# ── Figure savers ──────────────────────────────────────────────────────────────
def save_fig(fig, path, dpi):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_datasets(sorted_ds, datasets, splits, title, dpi, out_path):
    ds_rows = []
    for name, v in sorted_ds:
        ax_  = v.get("axial", 0)
        sag_ = v.get("sagittal", 0)
        ds_rows.append((name, fmt(v["total"]),
                        fmt(ax_) if ax_ else "—",
                        fmt(sag_) if sag_ else "—"))
    grand_total = sum(v["total"]            for v in datasets.values())
    grand_ax    = sum(v.get("axial", 0)    for v in datasets.values())
    grand_sag   = sum(v.get("sagittal", 0) for v in datasets.values())
    ds_rows.append(("TOTAL", fmt(grand_total), fmt(grand_ax), fmt(grand_sag)))

    n = len(ds_rows)
    fig_h = max(5, 0.44 * n + 1.2)
    fig = plt.figure(figsize=(10, fig_h), facecolor=BG)
    fig.suptitle(f"{title} — Datasets", fontsize=14, fontweight="bold", color=TXT, y=0.99)

    ax = fig.add_axes([0.02, 0.02, 0.96, 0.94])
    draw_table(ax,
               ["Dataset", "Slices (total)", "Axial", "Sagittal"],
               ds_rows,
               col_widths=[3.8, 1.5, 1.2, 1.2],
               alignments=["left", "right", "right", "right"])
    ax.plot([], [], "o", color=AX_COL,  label="Axial",    markersize=6)
    ax.plot([], [], "o", color=SAG_COL, label="Sagittal", markersize=6)
    ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=TXT)

    split_str = "  |  ".join(f"{k}: {fmt(v)}" for k, v in sorted(splits.items()))
    split_str += f"  |  Total: {fmt(sum(splits.values()))}"
    fig.text(0.5, 0.005, split_str, ha="center", fontsize=7.5, color="#555555")

    save_fig(fig, out_path, dpi)


def plot_orientation(orientations, splits, title, dpi, out_path):
    n_axial = orientations.get("axial", 0)
    n_sag   = orientations.get("sagittal", 0)

    fig, ax = plt.subplots(figsize=(6, 5.5), facecolor=BG)
    ax.set_facecolor(BG)
    fig.suptitle(f"{title} — Orientation", fontsize=14, fontweight="bold", color=TXT)

    pie_vals   = [v for v in [n_axial, n_sag] if v > 0]
    pie_labels = ([f"Axial\n{fmt(n_axial)}"] if n_axial else []) + \
                 ([f"Sagittal\n{fmt(n_sag)}"] if n_sag else [])
    pie_colors = ([AX_COL] if n_axial else []) + ([SAG_COL] if n_sag else [])

    wedges, texts, autotexts = ax.pie(
        pie_vals, labels=pie_labels, colors=pie_colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"color": TXT, "fontsize": 11},
        wedgeprops={"edgecolor": "white", "linewidth": 2.5})
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
        at.set_fontsize(11)

    split_str = "  |  ".join(f"{k}: {fmt(v)}" for k, v in sorted(splits.items()))
    split_str += f"  |  Total: {fmt(sum(splits.values()))}"
    ax.annotate(split_str, xy=(0.5, -0.04), xycoords="axes fraction",
                ha="center", fontsize=8.5, color="#555555")

    save_fig(fig, out_path, dpi)


def plot_contrasts(sorted_ct, title, dpi, out_path):
    ct_names = [c[0] for c in sorted_ct]
    ct_vals  = [c[1] for c in sorted_ct]
    palette  = ["#2d3a8c","#1976d2","#0097a7","#388e3c",
                "#f57c00","#7b1fa2","#c62828","#00695c","#558b2f"]

    fig_h = max(4, 0.55 * len(ct_names) + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_h), facecolor=BG)
    ax.set_facecolor(CELL_D)
    fig.suptitle(f"{title} — Contrasts", fontsize=14, fontweight="bold", color=TXT)

    colors = [palette[i % len(palette)] for i in range(len(ct_names))]
    bars   = ax.barh(ct_names, ct_vals, color=colors,
                     edgecolor="white", linewidth=0.8, height=0.6)
    ax.set_xlim(0, max(ct_vals) * 1.22)
    ax.set_xlabel("Slice count", color=TXT, fontsize=10)
    for bar, val in zip(bars, ct_vals):
        ax.text(val + max(ct_vals) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                fmt(val), va="center", fontsize=9, color=TXT)
    ax.tick_params(axis="x", colors="#777777", labelsize=9)
    ax.tick_params(axis="y", colors=TXT, labelsize=10)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_edgecolor("#cccccc")
    ax.xaxis.label.set_color("#777777")
    fig.tight_layout()

    save_fig(fig, out_path, dpi)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Summarize an out_raw_numero directory into 3 separate PNGs.")
    parser.add_argument("--input-dir", required=True,
                        help="Root of the output directory (contains train/, val/, ...)")
    parser.add_argument("--output-dir", default=".",
                        help="Directory where the 3 PNGs are saved (default: current dir)")
    parser.add_argument("--prefix", default="summary",
                        help="Filename prefix (default: 'summary')")
    parser.add_argument("--title", default=None,
                        help="Custom title (default: directory name)")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    root     = Path(args.input_dir).expanduser().resolve()
    out_dir  = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    title    = args.title or root.name

    print(f"Scanning {root} ...")
    datasets, contrasts, splits, orientations = parse_files(root)
    print(f"  Found {sum(splits.values()):,} files across splits: {dict(splits)}")

    sorted_ds = sorted(datasets.items(), key=lambda kv: kv[1]["total"], reverse=True)
    sorted_ct = sorted(contrasts.items(), key=lambda kv: kv[1], reverse=True)

    print("Generating figures ...")
    plot_datasets(  sorted_ds, datasets, splits, title, args.dpi,
                    out_dir / f"{args.prefix}_datasets.png")
    plot_orientation(orientations, splits, title, args.dpi,
                    out_dir / f"{args.prefix}_orientation.png")
    plot_contrasts( sorted_ct, title, args.dpi,
                    out_dir / f"{args.prefix}_contrasts.png")


if __name__ == "__main__":
    main()
