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
    parser.add_argument("--set", nargs="*", default=[],
                        help="Key=value overrides, e.g. --set dilation_radius=4 task=nfn_dil4")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = OmegaConf.load(config_path)
    if args.set:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.set))
    main(config)
