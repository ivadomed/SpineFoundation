# SpineMAE Downstream Training

This module trains a downstream task by freezing the SpineMAE encoder and learning a small head on top using labeled data.

## Features
- Loads and freezes SpineMAE encoder from `mae_training/ckpt/best.ckpt`
- Supports classification (binary/multi-class) and regression
- Config-driven datasets, transforms, and training parameters
- Checkpointing and metrics (accuracy, F1, MAE/MSE)

## Quick Start

1. Edit `config.json` to point to your dataset and labels.
2. Run training:

```bash
python -m SpineFoundation.downstream_training.train --config SpineFoundation/downstream_training/config.json
```

## Expected Data
- Images: path to folders or a CSV listing image files
- Labels: CSV with `image_path,label`

## Notes
- Encoder weights are loaded from `SpineFoundation/mae_training/ckpt/best.ckpt`.
- Ensure dependencies from SpineFoundation `requirements.txt` are installed.