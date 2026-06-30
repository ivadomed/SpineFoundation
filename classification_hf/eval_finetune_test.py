"""
Evaluate fine-tuned unfrozen-backbone models on the RSNA test set.

Usage:
    python -m classification_hf.eval_finetune_test \
        [--tune_dir outputs_cls/tune_scs_finetune] \
        [--data_dir ~/data/RSNA_patches_scs] \
        [--fold_csv ~/fold_split_RSNA.json] \
        [--output_md results_scs_finetune_test.md]
"""
import argparse
import glob
from functools import partial
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.special import softmax as sp_softmax
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                              roc_auc_score)
from transformers import AutoImageProcessor, Dinov2Model

from .dataset import load_test_dataset, preprocess_function
from .model import MaskedBackboneClassifier

MODEL_NAME = ("/home/ge.polymtl.ca/p123239/.cache/huggingface/hub/"
              "models--raidium--curia/snapshots/"
              "9657dc56276bc6c9503ef6f8d060879c8bee482f")
CROP_CM    = 4.0
NUM_CLASSES = 3
BATCH_SIZE  = 32
REGIMES = ["50", "100", "200", "300", "400", "500", "750", "all"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tune_dir",  default="outputs_cls/tune_scs_finetune")
    p.add_argument("--data_dir",  default="/home/ge.polymtl.ca/p123239/data/RSNA_patches_scs")
    p.add_argument("--fold_csv",  default="/home/ge.polymtl.ca/p123239/fold_split_RSNA.json")
    p.add_argument("--output_md", default="results_scs_finetune_test.md")
    p.add_argument("--gpu",       type=int, default=1)
    return p.parse_args()


def _load_model(run_dir: Path, device: torch.device) -> MaskedBackboneClassifier:
    backbone    = Dinov2Model.from_pretrained(MODEL_NAME, trust_remote_code=True)
    hidden_size = backbone.config.hidden_size
    model       = MaskedBackboneClassifier(backbone, hidden_size, NUM_CLASSES)
    state_dict  = load_file(str(run_dir / "model.safetensors"))
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model


def _run_inference(model, test_ds, processor, device):
    """Returns (logits np.ndarray, labels np.ndarray)."""
    _prep = partial(preprocess_function, processor=processor, crop_cm=CROP_CM)
    test_ds = test_ds.map(_prep, batched=True, batch_size=BATCH_SIZE, num_proc=None)

    columns = ["pixel_values", "labels"]
    if "mask" in test_ds.column_names:
        columns.append("mask")
    test_ds.set_format(type="torch", columns=columns)

    all_logits, all_labels = [], []
    with torch.no_grad():
        for i in range(0, len(test_ds), BATCH_SIZE):
            batch = test_ds[i:i + BATCH_SIZE]
            pv = batch["pixel_values"].to(device)
            lbl = batch["labels"]
            mask = batch.get("mask")
            if mask is not None:
                mask = mask.to(device)
            out = model(pixel_values=pv, mask=mask)
            all_logits.append(out["logits"].cpu().float().numpy())
            all_labels.append(lbl.numpy() if isinstance(lbl, torch.Tensor) else np.array(lbl))

    return np.vstack(all_logits), np.concatenate(all_labels)


def _metrics(logits, labels):
    probs = sp_softmax(logits.astype(np.float64), axis=1)
    probs_c = np.clip(probs, 1e-7, 1.0)
    probs_c /= probs_c.sum(1, keepdims=True)
    wbce = float(np.mean(
        np.array([1., 2., 4.])[labels] *
        (-np.log(probs_c[np.arange(len(labels)), labels]))
    ))
    auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    acc = accuracy_score(labels, logits.argmax(1))
    qwk = cohen_kappa_score(labels, logits.argmax(1), weights="quadratic")
    return {"auc": auc, "acc": acc, "qwk": qwk, "wbce": wbce}


def main():
    args   = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    test_ds   = load_test_dataset(args.data_dir, args.fold_csv)
    n_test    = len(test_ds)
    print(f"Test samples: {n_test}")

    results = []
    for regime in REGIMES:
        regime_dir = Path(args.tune_dir) / f"regime_{regime}_split_1_set" / "final"
        runs = sorted(glob.glob(str(regime_dir / "scs__*")))
        if not runs:
            print(f"[regime_{regime}] no final run found — skipping")
            continue
        run_dir = Path(runs[-1])
        st_path = run_dir / "model.safetensors"
        if not st_path.exists():
            print(f"[regime_{regime}] model.safetensors missing — skipping")
            continue

        print(f"\n[regime_{regime}] loading model from {run_dir.name} …")
        model   = _load_model(run_dir, device)
        logits, labels = _run_inference(model, test_ds, processor, device)
        m = _metrics(logits, labels)
        results.append((regime, m, len(labels)))

        print(f"  AUC={m['auc']:.4f}  Acc={m['acc']:.4f}  "
              f"QWK={m['qwk']:.4f}  WBCE={m['wbce']:.4f}  (n={len(labels)})")

        # Save predictions
        np.savez(str(run_dir / "test_predictions.npz"), logits=logits, labels=labels)

        del model
        torch.cuda.empty_cache()

    # Write markdown
    lines = [
        "# SCS Finetune (unfrozen backbone) — Test Set Results\n",
        "| Régime | AUC | Acc | QWK | WBCE | n_test |",
        "|--------|-----|-----|-----|------|--------|",
    ]
    for regime, m, n in results:
        lines.append(
            f"| {regime} | {m['auc']:.4f} | {m['acc']:.4f} | "
            f"{m['qwk']:.4f} | {m['wbce']:.4f} | {n} |"
        )

    lines += [
        "",
        "## Référence (frozen backbone, crop 4cm)",
        "| Modèle | AUC | QWK | WBCE |",
        "|--------|-----|-----|------|",
        "| Resnet (regime_all) | 0.9545 | 0.701 | 0.4280 |",
        "| CLS    (regime_all) | 0.9394 | 0.645 | 0.5019 |",
    ]

    md_path = Path(args.tune_dir).parent.parent / args.output_md
    md_path.write_text("\n".join(lines) + "\n")
    print(f"\nRésultats écrits dans {md_path}")


if __name__ == "__main__":
    main()
