import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExtractConfig:
    model_dir: str
    data_dir: str
    output_dir: str
    image_size: int = 512
    batch_size: int = 64
    num_workers: int = 8
    amp: bool = True
    val_split: float = 0.2
    seed: int = 42


@dataclass
class TrainConfig:
    features_dir: str
    output_dir: str = "outputs_cls"
    epochs: int = 200
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 42
    use_class_weights: bool = True
    hidden_dim: int = 256
    dropout: float = 0.2
    use_wandb: bool = False
    wandb_project: str = "spine-cls"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str = "online"


def parse_extract_args() -> ExtractConfig:
    parser = argparse.ArgumentParser(description="Extract backbone features for classification")
    parser.add_argument("--model_dir", required=True, help="Path to HF backbone directory")
    parser.add_argument("--data_dir", required=True, help="Root directory with one subfolder per class")
    parser.add_argument("--output_dir", required=True, help="Directory to save train.npz and val.npz")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.set_defaults(amp=True)
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return ExtractConfig(**vars(args))


def parse_train_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train classification head on pre-extracted features")
    parser.add_argument("--features_dir", required=True, help="Directory with train.npz and val.npz")
    parser.add_argument("--output_dir", default="outputs_cls")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.set_defaults(use_class_weights=True)
    parser.add_argument("--use_class_weights", dest="use_class_weights", action="store_true")
    parser.add_argument("--no_class_weights", dest="use_class_weights", action="store_false")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.set_defaults(use_wandb=False)
    parser.add_argument("--wandb", dest="use_wandb", action="store_true")
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    parser.add_argument("--wandb_project", default="spine-cls")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])

    args = parser.parse_args()
    wandb_run_name = args.wandb_run_name or (Path(args.output_dir).name or "run")

    return TrainConfig(
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        use_class_weights=args.use_class_weights,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=wandb_run_name,
        wandb_mode=args.wandb_mode,
    )
