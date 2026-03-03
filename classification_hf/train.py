"""
Entry point — mirrors curia usage:

    python -m classification_hf.train --config classification_hf/configs/rsna_neural_foraminal_narrowing.yaml
"""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from .trainer import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classifier with OmegaConf config")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = OmegaConf.load(config_path)
    main(config)
