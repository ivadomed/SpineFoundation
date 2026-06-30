"""
Re-extrait les patch tokens pour toutes les images BrnoSpine (train + test)
et les sauvegarde dans un sous-dossier de segmentation_hf/data/.

Usage:
    # Curia (défaut)
    python -m segmentation_hf.cache_brno_features

    # DINOv3
    python -m segmentation_hf.cache_brno_features \
        --model_dir /path/to/dinov3-vitl16 \
        --out_dir   segmentation_hf/data/brno_npz_dinov3
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoImageProcessor, AutoModel

NNUNET_RAW = Path("/home/ge.polymtl.ca/p123239/data/nnUNetv2/nnUNet_raw/Dataset026_BrnoSpineAll")
IMAGES_TR  = NNUNET_RAW / "imagesTr"
LABELS_TR  = NNUNET_RAW / "labelsTr"
IMAGES_TS  = NNUNET_RAW / "imagesTs"
LABELS_TS  = NNUNET_RAW / "labelsTs"

DEFAULT_OUT_DIR   = Path(__file__).parent / "data" / "brno_npz"
DEFAULT_MODEL_DIR = str(Path.home() / ".cache/huggingface/hub/models--raidium--curia/snapshots/9657dc56276bc6c9503ef6f8d060879c8bee482f")

BATCH_SIZE  = 32
NUM_WRITERS = 8
MASK_SIZE   = 512


def load_image_and_mask(img_path: Path, mask_path: Path, img_mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with Image.open(img_path) as img:
        img_pil = img.convert(img_mode)
    with Image.open(mask_path) as msk:
        msk_r = msk.convert("L").resize((MASK_SIZE, MASK_SIZE), Image.NEAREST)
    img_raw  = np.array(img_pil.convert("L").resize((MASK_SIZE, MASK_SIZE), Image.BILINEAR), dtype=np.float32)
    mask     = (np.array(msk_r, dtype=np.float32) > 0).astype(np.float32)
    img_arr  = np.array(img_pil)  # (H,W) pour L, (H,W,3) pour RGB
    return img_raw, mask, img_arr


def save_npz(out_path: Path, img_raw: np.ndarray, mask: np.ndarray, tokens: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(str(tmp), slice=img_raw, mask=mask, patch_tokens=tokens)
    tmp.rename(out_path)


def collect_pairs(images_dir: Path, labels_dir: Path, overwrite: bool, out_dir: Path) -> list[tuple[Path, Path, Path]]:
    pairs = []
    for img_path in sorted(images_dir.glob("*.png")):
        stem = img_path.stem
        base = stem[:-5] if stem.endswith("_0000") else stem
        mask_path = labels_dir / f"{base}.png"
        out_path  = out_dir / f"{base}.npz"
        if not mask_path.exists():
            print(f"[WARN] Mask introuvable : {mask_path}")
            continue
        if not overwrite and out_path.exists():
            continue
        pairs.append((img_path, mask_path, out_path))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite",  action="store_true")
    parser.add_argument("--model_dir",  default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out_dir",    default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    all_pairs  = collect_pairs(IMAGES_TR, LABELS_TR, args.overwrite, out_dir)
    all_pairs += collect_pairs(IMAGES_TS, LABELS_TS, args.overwrite, out_dir)

    total_imgs = len(list(IMAGES_TR.glob("*.png"))) + len(list(IMAGES_TS.glob("*.png")))
    print(f"Total images    : {total_imgs}")
    print(f"À traiter       : {len(all_pairs)}  (déjà cachés : {total_imgs - len(all_pairs)})")
    print(f"Modèle          : {args.model_dir}")
    print(f"Sortie          : {out_dir}")
    if not all_pairs:
        print("Rien à faire.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device          : {device}")

    config    = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    processor = AutoImageProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    backbone  = AutoModel.from_pretrained(args.model_dir, config=config, trust_remote_code=True)
    backbone.to(device).eval()

    num_channels       = int(getattr(config, "num_channels", 3))
    img_mode           = "L" if num_channels == 1 else "RGB"
    patch_size         = int(getattr(config, "patch_size", 16))
    image_size         = int(getattr(config, "image_size", 224))
    num_register_tokens = int(getattr(config, "num_register_tokens", 0))
    n_prefix           = 1 + num_register_tokens  # CLS + registers

    print(f"num_channels    : {num_channels}  ({img_mode})")
    print(f"image_size      : {image_size}")
    print(f"patch_size      : {patch_size}")
    print(f"num_registers   : {num_register_tokens}  (n_prefix={n_prefix})")

    errors: list[str] = []
    prev_futures: list = []

    def _wait(futures):
        for f in futures:
            try:
                f.result()
            except Exception as e:
                errors.append(str(e))

    with ThreadPoolExecutor(max_workers=NUM_WRITERS) as pool:
        with tqdm(total=len(all_pairs), desc="cache patch_tokens", unit="img") as pbar:
            for start in range(0, len(all_pairs), BATCH_SIZE):
                batch = all_pairs[start : start + BATCH_SIZE]

                loaded   = [load_image_and_mask(ip, mp, img_mode) for ip, mp, _ in batch]
                imgs_raw = [x[0] for x in loaded]
                masks    = [x[1] for x in loaded]
                imgs_arr = [x[2] for x in loaded]

                inputs = processor(images=imgs_arr, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device)

                with torch.no_grad():
                    encoder = backbone.vision_model if config.model_type == "clip" else backbone
                    out    = encoder(pixel_values=pixel_values, return_dict=True)
                    tokens = out.last_hidden_state
                    tokens = tokens[:, n_prefix:, :]  # strip CLS + register tokens
                    tokens_np = tokens.cpu().float().numpy()

                _wait(prev_futures)
                prev_futures = [
                    pool.submit(save_npz, out_path, imgs_raw[i], masks[i], tokens_np[i])
                    for i, (_, _, out_path) in enumerate(batch)
                ]
                pbar.update(len(batch))

        _wait(prev_futures)

    if errors:
        print(f"\n[WARN] {len(errors)} erreurs d'écriture :")
        for e in errors[:5]:
            print(f"  {e}")

    print(f"\nNPZ sauvegardés dans : {out_dir}")


if __name__ == "__main__":
    main()
