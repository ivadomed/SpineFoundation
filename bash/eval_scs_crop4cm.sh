#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=1

PYTHON=/home/ge.polymtl.ca/p123239/.conda/envs/FM/bin/python
REPO=/home/ge.polymtl.ca/p123239/SpineFoundation

cd "$REPO"

"$PYTHON" - << 'PYEOF'
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, cohen_kappa_score
import sys
sys.path.insert(0, ".")

from classification_hf.dataset import load_test_dataset, CropTokenDataset
from classification_hf.model import TokenGridClassifier, Classifier

DATA_DIR  = "/home/ge.polymtl.ca/p123239/data/RSNA_patches_scs"
FOLD_CSV  = "/home/ge.polymtl.ca/p123239/fold_split_RSNA.json"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH     = 256

# ── Test dataset ──────────────────────────────────────────────────────────────
hf_test = load_test_dataset(DATA_DIR, FOLD_CSV)
print(f"Test samples: {len(hf_test['path'])}", flush=True)

def latest_run(fold_dir):
    runs = sorted(Path(fold_dir).glob("scs__*"))
    return runs[-1] if runs else None

CLASS_WEIGHTS = np.array([1.0, 2.0, 4.0])

def metrics(logits, labels):
    from scipy.special import softmax as sp_softmax
    probs = np.clip(sp_softmax(logits.astype(np.float64), axis=1), 1e-7, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    preds = logits.argmax(axis=1)
    auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    acc = accuracy_score(labels, preds)
    qwk = cohen_kappa_score(labels, preds, weights="quadratic")
    ce = float(np.mean(CLASS_WEIGHTS[labels] * (-np.log(probs[np.arange(len(labels)), labels]))))
    return auc, acc, qwk, ce

# ── Resnet ────────────────────────────────────────────────────────────────────
print("\n=== Resnet (TokenGridClassifier) — test set ===")
print(f"{'Regime':<8} {'AUC':>7} {'Acc':>7} {'QWK':>7} {'CE':>7}")
print("-"*42)

TOKEN_KEY = "patch_tokens_curia_crop4cm"
test_ds_resnet = CropTokenDataset(hf_test["path"], hf_test["target"],
                                  token_key=TOKEN_KEY, preload=True)
tokens_gpu = test_ds_resnet.tokens.to(DEVICE)
labels_arr = np.array(test_ds_resnet.labels)

REGIMES = ["50","100","200","300","400","500","750","1000","all"]

r_aucs, r_accs, r_qwks, r_ces = [], [], [], []
for regime in REGIMES:
    fold = f"regime_{regime}_split_1_set"
    run = latest_run(f"outputs_cls/tune_scs_resnet/{fold}/final")
    if run is None:
        print(f"  {regime:<6}  MISSING"); continue
    ckpt = torch.load(run / "head.pt", map_location="cpu", weights_only=True)
    sd   = ckpt["state_dict"]
    hidden = sd["proj.0.weight"].shape[1]
    pd_    = sd["proj.0.weight"].shape[0]
    nb_    = max((int(k.split(".")[1]) for k in sd if k.startswith("blocks.")), default=-1) + 1
    model  = TokenGridClassifier(hidden, 3, proj_dim=pd_, n_blocks=nb_)
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE).eval()
    all_logits = []
    with torch.no_grad():
        for j in range(0, len(test_ds_resnet), BATCH):
            pv  = tokens_gpu[j:j+BATCH].float()
            lbl = torch.tensor(labels_arr[j:j+BATCH], device=DEVICE)
            all_logits.append(model(pv, lbl)["logits"].cpu())
    logits = torch.cat(all_logits).numpy()
    auc, acc, qwk, ce = metrics(logits, labels_arr)
    r_aucs.append(auc); r_accs.append(acc); r_qwks.append(qwk); r_ces.append(ce)
    print(f"  {regime:<6}  {auc:.4f}  {acc:.4f}  {qwk:.4f}  {ce:.4f}")
print("-"*38)
print(f"  {'μ':<3}  {np.mean(r_aucs):.4f}  {np.mean(r_accs):.4f}  {np.mean(r_qwks):.4f}  {np.mean(r_ces):.4f}")
print(f"  {'σ':<3}  {np.std(r_aucs):.4f}  {np.std(r_accs):.4f}  {np.std(r_qwks):.4f}  {np.std(r_ces):.4f}")

# ── CLS linear ────────────────────────────────────────────────────────────────
print("\n=== CLS linear — test set ===")
print(f"{'Regime':<8} {'AUC':>7} {'Acc':>7} {'QWK':>7} {'CE':>7}")
print("-"*42)

CLS_KEY = "cls_token_curia_crop4cm"
test_ds_cls = CropTokenDataset(hf_test["path"], hf_test["target"],
                               token_key=CLS_KEY, preload=True)
cls_gpu    = test_ds_cls.tokens.to(DEVICE)

c_aucs, c_accs, c_qwks, c_ces = [], [], [], []
for regime in REGIMES:
    fold = f"regime_{regime}_split_1_set"
    run = latest_run(f"outputs_cls/tune_scs_cls/{fold}/final")
    if run is None:
        print(f"  {regime:<6}  MISSING"); continue
    ckpt   = torch.load(run / "head.pt", map_location="cpu", weights_only=True)
    sd     = ckpt["classifier"]
    hidden = sd["weight"].shape[1]
    model  = Classifier(hidden, 3)
    model.linear.load_state_dict(sd)
    model.to(DEVICE).eval()
    all_logits = []
    with torch.no_grad():
        for j in range(0, len(test_ds_cls), BATCH):
            pv  = cls_gpu[j:j+BATCH].float()
            lbl = torch.tensor(labels_arr[j:j+BATCH], device=DEVICE)
            all_logits.append(model(pv, lbl)["logits"].cpu())
    logits = torch.cat(all_logits).numpy()
    auc, acc, qwk, ce = metrics(logits, labels_arr)
    c_aucs.append(auc); c_accs.append(acc); c_qwks.append(qwk); c_ces.append(ce)
    print(f"  {regime:<6}  {auc:.4f}  {acc:.4f}  {qwk:.4f}  {ce:.4f}")
print("-"*38)
print(f"  {'μ':<3}  {np.mean(c_aucs):.4f}  {np.mean(c_accs):.4f}  {np.mean(c_qwks):.4f}  {np.mean(c_ces):.4f}")
print(f"  {'σ':<3}  {np.std(c_aucs):.4f}  {np.std(c_accs):.4f}  {np.std(c_qwks):.4f}  {np.std(c_ces):.4f}")
PYEOF

echo "==> Done"
