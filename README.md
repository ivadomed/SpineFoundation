# SpineFoundation

DINOv2-based framework for spine MRI analysis: self-supervised pretraining on medical images + downstream classification and segmentation.

Forked from [mselmangokmen/dinov2-training-HF](https://github.com/mselmangokmen/dinov2-training-HF).

---

## Overview

```
NIfTI volumes (BIDS)
    ↓  slice_extraction/
2D PNG / NPZ slices
    ↓  (optional) bash/runcuria.sh
Fine-tuned DINOv2 backbone
    ↓  downstream tasks
Classification (RSNA)  /  Segmentation (BrnoSpine)
```

---

## Installation

```bash
conda create -n dino python=3.11 -y
conda activate dino
pip install torch==2.8.0+cu129 torchvision==0.23.0+cu129 --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
```

Install Curia pretrained weights:

```bash
python install_curia.py
```

---

## Step 1 — Extract 2D slices from NIfTI volumes

See [`slice_extraction/README.md`](slice_extraction/README.md) for full documentation.

Quick example:

```bash
python slice_extraction/05_run_pipeline.py \
    --input-images /data/images \
    --input-labels /data/labels \
    --label-suffix _seg \
    --work-root /data/work
```

Output: `work-root/02_final/image/{train,val}/` and `work-root/02_final/label/{train,val}/`.

Data must follow an ImageNet-like layout (one sub-folder per class) for classification, or a flat directory for segmentation:

```
data/
├── train/
│   ├── n0000/
│   │   ├── n0000_000001.JPEG
│   │   └── n0000_000002.JPEG
│   └── n0001/
│       └── n0001_000001.JPEG
└── val/
    ├── n0000/
    │   └── n0000_000101.JPEG
    └── n0001/
        └── n0001_000101.JPEG
```

Note: class names (`n000X`) do not represent real categories — you can put everything in the same class for unlabelled pretraining.

---

## Step 2 — (Optional) Fine-tune the backbone

Fine-tune Curia (DINOv2) with self-supervised learning on your own data:

```bash
# 1. Edit configs/dino/configcuria.yaml — set data path and checkpoint
# 2. Launch on 2 GPUs:
bash bash/runcuria.sh
```

Skip this step to use the pretrained Curia weights directly (`python install_curia.py`).

---

## Step 3 — Downstream tasks

### Classification (RSNA Lumbar Spine)

Severity grading (Normal / Moderate / Severe) for spinal canal stenosis, neural foraminal narrowing, and subarticular stenosis.

→ See [`classification_hf/README.md`](classification_hf/README.md)

### Segmentation (BrnoSpine / custom)

Binary segmentation with a frozen DINOv2 backbone and a lightweight head.

→ See [`segmentation_hf/README.md`](segmentation_hf/README.md)

---

## Project structure

```
SpineFoundation/
├── configs/              ← Training configs (backbone pretraining + model definitions)
│   ├── dino/             ←   DINOv2 self-supervised training
│   └── models/           ←   Model registry (models.json)
├── slice_extraction/     ← 2D slice extraction from NIfTI volumes
├── classification_hf/    ← RSNA classification downstream task
├── segmentation_hf/      ← Segmentation downstream task
├── bash/                 ← Launch scripts (runcuria, experiments, eval)
├── utils/                ← Shared utilities (logging, checkpointing, distributed)
├── vit_models/           ← ViT architecture
├── train_dino.py         ← Backbone pretraining entry point
├── install_curia.py      ← Download Curia pretrained weights
└── requirements.txt
```

---

## Backbone format

All downstream tasks expect a HuggingFace-compatible checkpoint directory containing:
- `config.json` with `hidden_size` (or `embed_dim`) and `patch_size`
- `model.safetensors` (or `pytorch_model.bin`)
- `preprocessor_config.json`

Compatible with `raidium/curia` and any fine-tuned DINOv2 checkpoint.
