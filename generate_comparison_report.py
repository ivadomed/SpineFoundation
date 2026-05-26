#!/usr/bin/env python3
"""Generate a Curia vs DINOv3 comparison PDF report."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings("ignore")

CURIA_DIR  = Path("analysis_output_ax")
DINO3_DIR  = Path("analysis_output_dinov3_ax")
OUT_PDF    = Path("analysis_output_dinov3_ax/curia_vs_dinov3_report.pdf")

CURIA_COLOR = "#2166ac"
DINO3_COLOR = "#d6604d"
ATTR_LABELS = {
    "contrast_type": "Contrast type",
    "body_part":     "Body part",
    "pathology":     "Pathology",
    "dataset":       "Dataset",
    "spacing_bin":   "Spacing bin",
}


def load(d, suffix):
    return pd.read_csv(d / f"linear_probing_results_{suffix}.csv")


def bar_comparison(ax, curia_df, dino3_df, title, show_chance=True):
    attrs = curia_df["attribute"].tolist()
    x = np.arange(len(attrs))
    w = 0.32

    bars_c = ax.bar(x - w/2, curia_df["bal_acc_mean"], w,
                    yerr=curia_df["bal_acc_std"], capsize=4,
                    color=CURIA_COLOR, alpha=0.85, label="Curia")
    bars_d = ax.bar(x + w/2, dino3_df["bal_acc_mean"], w,
                    yerr=dino3_df["bal_acc_std"], capsize=4,
                    color=DINO3_COLOR, alpha=0.85, label="DINOv3-ViT-L")

    if show_chance and "chance" in curia_df.columns:
        for i, row in curia_df.iterrows():
            ax.hlines(row["chance"], i - 0.5, i + 0.5,
                      colors="grey", linestyles=":", linewidth=1.2)

    # delta annotation above each pair
    for i, (_, rc) in enumerate(curia_df.iterrows()):
        rd = dino3_df.iloc[i]
        delta = rd["bal_acc_mean"] - rc["bal_acc_mean"]
        color = DINO3_COLOR if delta > 0 else CURIA_COLOR
        sign  = "+" if delta >= 0 else ""
        ypos  = max(rc["bal_acc_mean"] + rc.get("bal_acc_std", 0),
                    rd["bal_acc_mean"] + rd.get("bal_acc_std", 0)) + 0.015
        ax.text(i, ypos, f"{sign}{delta:.3f}", ha="center", va="bottom",
                fontsize=7.5, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([ATTR_LABELS.get(a, a) for a in attrs],
                       rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Balanced accuracy (5-fold CV)", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # chance legend
    if show_chance:
        ax.hlines([], [], [], colors="grey", linestyles=":", linewidth=1.2,
                  label="chance level")
        ax.legend(fontsize=8)


def delta_heatmap(ax, curia_cls, dino3_cls, curia_pm, dino3_pm):
    attrs = [ATTR_LABELS.get(a, a) for a in curia_cls["attribute"]]
    cls_delta  = (dino3_cls["bal_acc_mean"].values  - curia_cls["bal_acc_mean"].values)
    pm_delta   = (dino3_pm["bal_acc_mean"].values   - curia_pm["bal_acc_mean"].values)
    data = np.vstack([cls_delta, pm_delta])

    im = ax.imshow(data, cmap="RdBu", vmin=-0.08, vmax=0.08, aspect="auto")
    plt.colorbar(im, ax=ax, label="DINOv3 − Curia (bal. acc.)", fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(attrs))); ax.set_xticklabels(attrs, rotation=20, ha="right", fontsize=9)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["CLS token", "Patch mean"], fontsize=9)
    ax.set_title("Δ balanced accuracy: DINOv3 − Curia  (blue = Curia better, red = DINOv3 better)",
                 fontsize=10, fontweight="bold")

    for i in range(2):
        vals = cls_delta if i == 0 else pm_delta
        for j, v in enumerate(vals):
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if abs(v) > 0.04 else "black")


def model_info_table(ax):
    ax.axis("off")
    info = [
        ["", "Curia", "DINOv3 ViT-Large"],
        ["Backbone",        "DINOv2-based (custom)", "DINOv3 ViT-L/16"],
        ["Parameters",      "~307M",                 "~307M"],
        ["Training data",   "Medical images",        "LVD-1689M (natural)"],
        ["Input size",      "512 × 512",             "224 × 224"],
        ["Channels",        "1 (grayscale)",         "3 (RGB replicated)"],
        ["Resize",          "Bicubic antialias",     "Bicubic"],
        ["Normalization",   "z-score per image",     "ImageNet stats"],
        ["Register tokens", "0",                     "4"],
        ["Embedding dim",   "1024",                  "1024"],
    ]
    tbl = ax.table(cellText=info[1:], colLabels=info[0],
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#e0e0e0")
            cell.set_text_props(fontweight="bold")
        elif c == 1:
            cell.set_facecolor("#ddeeff")
        elif c == 2:
            cell.set_facecolor("#fde8e4")
        if c == 0 and r > 0:
            cell.set_text_props(ha="left")
    ax.set_title("Model comparison overview", fontsize=11, fontweight="bold", pad=12)


def summary_table(ax, curia_cls, dino3_cls, curia_pm, dino3_pm):
    ax.axis("off")
    rows = []
    for i, attr in enumerate(curia_cls["attribute"]):
        rc, rd = curia_cls.iloc[i], dino3_cls.iloc[i]
        rpc, rpd = curia_pm.iloc[i], dino3_pm.iloc[i]
        winner_cls  = "Curia" if rc["bal_acc_mean"] > rd["bal_acc_mean"] + 0.005 else (
                       "DINOv3" if rd["bal_acc_mean"] > rc["bal_acc_mean"] + 0.005 else "≈ tie")
        winner_pm   = "Curia" if rpc["bal_acc_mean"] > rpd["bal_acc_mean"] + 0.005 else (
                       "DINOv3" if rpd["bal_acc_mean"] > rpc["bal_acc_mean"] + 0.005 else "≈ tie")
        rows.append([
            ATTR_LABELS.get(attr, attr),
            f"{rc['bal_acc_mean']:.3f}±{rc['bal_acc_std']:.3f}",
            f"{rd['bal_acc_mean']:.3f}±{rd['bal_acc_std']:.3f}",
            winner_cls,
            f"{rpc['bal_acc_mean']:.3f}±{rpc['bal_acc_std']:.3f}",
            f"{rpd['bal_acc_mean']:.3f}±{rpd['bal_acc_std']:.3f}",
            winner_pm,
        ])

    cols = ["Attribute",
            "Curia CLS", "DINOv3 CLS", "Winner (CLS)",
            "Curia patch", "DINOv3 patch", "Winner (patch)"]
    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#e0e0e0")
            cell.set_text_props(fontweight="bold")
        elif r > 0:
            if c == 3 or c == 6:
                val = rows[r-1][c]
                if val == "Curia":
                    cell.set_facecolor("#ddeeff")
                elif val == "DINOv3":
                    cell.set_facecolor("#fde8e4")
    ax.set_title("Summary table — Balanced accuracy (5-fold CV)", fontsize=11, fontweight="bold", pad=12)


def main():
    curia_cls = load(CURIA_DIR,  "cls")
    dino3_cls = load(DINO3_DIR,  "cls")
    curia_pm  = load(CURIA_DIR,  "patch_mean")
    dino3_pm  = load(DINO3_DIR,  "patch_mean")

    with PdfPages(OUT_PDF) as pdf:

        # ── Page 1 : title + model info + summary table ──────────────────────
        fig = plt.figure(figsize=(14, 10))
        fig.patch.set_facecolor("white")
        gs = gridspec.GridSpec(3, 1, figure=fig,
                               height_ratios=[0.12, 0.42, 0.46], hspace=0.55)

        # title
        ax_title = fig.add_subplot(gs[0])
        ax_title.axis("off")
        ax_title.text(0.5, 0.7, "Curia  vs  DINOv3 ViT-Large",
                      ha="center", va="center", fontsize=20, fontweight="bold",
                      transform=ax_title.transAxes)
        ax_title.text(0.5, 0.15,
                      "Linear probing — axial orientation — 5-fold stratified group CV",
                      ha="center", va="center", fontsize=11, color="grey",
                      transform=ax_title.transAxes)

        model_info_table(fig.add_subplot(gs[1]))
        summary_table(fig.add_subplot(gs[2]), curia_cls, dino3_cls, curia_pm, dino3_pm)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── Page 2 : bar charts CLS + patch mean ─────────────────────────────
        fig, axes = plt.subplots(2, 1, figsize=(13, 11))
        fig.patch.set_facecolor("white")
        bar_comparison(axes[0], curia_cls, dino3_cls,
                       "CLS token — balanced accuracy by attribute")
        bar_comparison(axes[1], curia_pm,  dino3_pm,
                       "Patch mean — balanced accuracy by attribute")
        plt.tight_layout(pad=2.5)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── Page 3 : delta heatmap ────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor("white")
        delta_heatmap(ax, curia_cls, dino3_cls, curia_pm, dino3_pm)
        plt.tight_layout(pad=2)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── Page 4 : CLS vs patch_mean within each model ─────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor("white")
        for ax, (cls_df, pm_df, name, color) in zip(axes, [
            (curia_cls, curia_pm, "Curia",          CURIA_COLOR),
            (dino3_cls, dino3_pm, "DINOv3 ViT-L",  DINO3_COLOR),
        ]):
            attrs = cls_df["attribute"].tolist()
            x = np.arange(len(attrs)); w = 0.32
            ax.bar(x - w/2, cls_df["bal_acc_mean"], w,
                   yerr=cls_df["bal_acc_std"], capsize=4,
                   color=color, alpha=0.85, label="CLS token")
            ax.bar(x + w/2, pm_df["bal_acc_mean"], w,
                   yerr=pm_df["bal_acc_std"], capsize=4,
                   color=color, alpha=0.45, label="Patch mean")
            ax.set_xticks(x)
            ax.set_xticklabels([ATTR_LABELS.get(a, a) for a in attrs],
                               rotation=20, ha="right", fontsize=9)
            ax.set_ylabel("Balanced accuracy", fontsize=9)
            ax.set_ylim(0, 1.1)
            ax.set_title(f"{name} — CLS vs patch mean", fontsize=11, fontweight="bold")
            ax.legend(fontsize=8)
            ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        plt.tight_layout(pad=2.5)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # metadata
        d = pdf.infodict()
        d["Title"]   = "Curia vs DINOv3 ViT-Large — Linear Probing Comparison"
        d["Author"]  = "analyze_embeddings.py"
        d["Subject"] = "Spine Foundation Model Evaluation"

    print(f"PDF saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
