import argparse
from dataclasses import dataclass


@dataclass
class TrainConfig:
    model_dir: str
    train_images: str
    train_masks: str
    val_images: str
    val_masks: str
    output_dir: str = "outputs_seg"
    image_size: int = 224
    tile_overlap: int = 56
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


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train segmentation head from HF-like checkpoint directory")

    parser.add_argument("--model_dir", type=str, required=True, help="Path to HF-like directory (config.json + model.safetensors)")
    parser.add_argument("--train_images", type=str, required=True, help="Path to train/images")
    parser.add_argument("--train_masks", type=str, required=True, help="Path to train/masks")
    parser.add_argument("--val_images", type=str, required=True, help="Path to val/images")
    parser.add_argument("--val_masks", type=str, required=True, help="Path to val/masks")

    parser.add_argument("--output_dir", type=str, default="outputs_seg")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--tile_overlap", type=int, default=56)
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

    args = parser.parse_args()
    amp = True
    if args.amp:
        amp = True
    if args.no_amp:
        amp = False

    return TrainConfig(
        model_dir=args.model_dir,
        train_images=args.train_images,
        train_masks=args.train_masks,
        val_images=args.val_images,
        val_masks=args.val_masks,
        output_dir=args.output_dir,
        image_size=args.image_size,
        tile_overlap=args.tile_overlap,
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
    )
