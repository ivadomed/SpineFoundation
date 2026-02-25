import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    model_dir: str
    train_images: str
    train_masks: str
    val_images: str
    val_masks: str
    output_dir: str = "outputs_seg"
    image_size: int = 224
    only_sagittal: bool = False
    tile_overlap_pct: float = 25.0
    tile_threshold: int = 512
    epochs: int = 50
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 8
    seed: int = 42
    amp: bool = True
    save_every: int = 1
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    use_wandb: bool = False
    wandb_project: str = "spine-seg"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str = "online"
    wandb_log_val_images: bool = True
    wandb_val_images_count: int = 4
    wandb_val_images_every: int = 1


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train segmentation head from HF-like checkpoint directory")

    parser.add_argument("--model_dir", type=str, required=True, help="Path to HF-like directory (config.json + model.safetensors)")
    parser.add_argument("--train_images", type=str, required=True, help="Path to train/images")
    parser.add_argument("--train_masks", type=str, required=True, help="Path to train/masks")
    parser.add_argument("--val_images", type=str, required=True, help="Path to val/images")
    parser.add_argument("--val_masks", type=str, required=True, help="Path to val/masks")

    parser.add_argument("--output_dir", type=str, default="outputs_seg")
    parser.add_argument("--image_size", type=int, default=224)
    parser.set_defaults(only_sagittal=False)
    parser.add_argument("--only_sagittal", dest="only_sagittal", action="store_true", help="Use only sagittal files")
    parser.add_argument("--all_planes", dest="only_sagittal", action="store_false", help="Use all files (default)")
    parser.add_argument("--tile_overlap_pct", type=float, default=25.0)
    parser.add_argument("--tile_threshold", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--amp", action="store_true", help="Enable mixed precision")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--save_every", type=int, default=1)

    parser.add_argument("--bce_weight", type=float, default=0.5)
    parser.add_argument("--dice_weight", type=float, default=0.5)

    parser.set_defaults(use_wandb=False)
    parser.add_argument("--wandb", dest="use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="spine-seg")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name (default: basename of output_dir)")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.set_defaults(wandb_log_val_images=True)
    parser.add_argument("--wandb_log_val_images", dest="wandb_log_val_images", action="store_true")
    parser.add_argument("--no_wandb_log_val_images", dest="wandb_log_val_images", action="store_false")
    parser.add_argument("--wandb_val_images_count", type=int, default=4)
    parser.add_argument("--wandb_val_images_every", type=int, default=1)

    args = parser.parse_args()
    amp = True
    if args.amp:
        amp = True
    if args.no_amp:
        amp = False

    wandb_run_name = args.wandb_run_name
    if wandb_run_name is None:
        wandb_run_name = Path(args.output_dir).name or "run"

    return TrainConfig(
        model_dir=args.model_dir,
        train_images=args.train_images,
        train_masks=args.train_masks,
        val_images=args.val_images,
        val_masks=args.val_masks,
        output_dir=args.output_dir,
        image_size=args.image_size,
        only_sagittal=args.only_sagittal,
        tile_overlap_pct=args.tile_overlap_pct,
        tile_threshold=args.tile_threshold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        amp=amp,
        save_every=args.save_every,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=wandb_run_name,
        wandb_mode=args.wandb_mode,
        wandb_log_val_images=args.wandb_log_val_images,
        wandb_val_images_count=args.wandb_val_images_count,
        wandb_val_images_every=args.wandb_val_images_every,
    )
