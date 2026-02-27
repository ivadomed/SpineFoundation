# THIS IS A FORK OF https://github.com/mselmangokmen/dinov2-training-HF

# DINOv2 Training Framework

## How to use it :

```bash
conda create -n dino python=3.11 -y
conda activate dino
pip install torch==2.8.0+cu129 torchvision==0.23.0+cu129 --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
```

## Install curia pretrained weights :

```bash
python install_curia.py
```

## Config files :

```text
SpineFoundation/
├── configs/
│   ├── dino/
│   │   ├── configcuria.yaml ← training config file
│   └── models/
│       ├── models.json ← model config file (see 'curia')
```

Specify your checkpoint path and data location there.

## Data

Data must be ImageNet-like:

```text
data/
├── train/
│   ├── n0000/
│   │   ├── n0000_000001.JPEG
│   │   ├── n0000_000002.JPEG
│   │   └── n0000_000003.JPEG
│   ├── n0001/
│   │   ├── n0001_000001.JPEG
│   │   └── n0001_000002.JPEG
│   ├── n0002/
│   │   └── n0002_000001.JPEG
│   └── n0003/
│       └── n0003_000001.JPEG
├── val/
│   ├── n0000/
│   │   └── n0000_000101.JPEG
│   ├── n0001/
│   │   └── n0001_000101.JPEG
```

Note: class `n000X` does not represent a real category but is mandatory (you can put everything in the same class).

You can extract data (with or without labels) from a 3D NIfTI volumes root folder using:
```text
slice_extraction/extract.sh
```
## Curia finetune 

```bash 
bash runcuria.sh
```

## Downstream tasks

You can train downstream tasks using :
```text
segmentation_hf/train.sh

