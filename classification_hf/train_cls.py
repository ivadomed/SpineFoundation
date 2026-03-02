"""
Train a classification head on pre-extracted backbone features.

Step 1 — extract features (run once):
    python -m classification_hf.extract_features \
        --model_dir /path/to/hf_checkpoint \
        --data_dir  /path/to/RSNA_patches_512 \
        --output_dir /path/to/features

Step 2 — train classifier (fast, no backbone inference):
    python -m classification_hf.train_cls \
        --features_dir /path/to/features \
        --output_dir   /path/to/outputs_cls
"""

from .config import parse_train_args
from .trainer import train

if __name__ == "__main__":
    cfg = parse_train_args()
    train(cfg)
