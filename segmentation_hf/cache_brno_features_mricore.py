"""
Extrait les patch tokens MRICore pour toutes les images BrnoSpine (train + test)
et les sauvegarde dans segmentation_hf/data/brno_npz_mricore/.

MRICore est basé sur SAM ViT-B : image_encoder() retourne (B, 256, 64, 64).
On reshape en (B, 4096, 256) pour le fast path NPZ.

Usage:
    python -m segmentation_hf.cache_brno_features_mricore [--overwrite]
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

MRICORE_REPO = Path("/home/ge.polymtl.ca/p123239/FM/models/mricore_repo")
MRICORE_CKPT = Path("/home/ge.polymtl.ca/p123239/FM/models/mricore/MRI_CORE_vitb.pth")
OUT_DIR      = Path(__file__).parent / "data" / "brno_npz_mricore"

NNUNET_RAW = Path("/home/ge.polymtl.ca/p123239/data/nnUNetv2/nnUNet_raw/Dataset026_BrnoSpineAll")
IMAGES_TR  = NNUNET_RAW / "imagesTr"
LABELS_TR  = NNUNET_RAW / "labelsTr"
IMAGES_TS  = NNUNET_RAW / "imagesTs"
LABELS_TS  = NNUNET_RAW / "labelsTs"

IMAGE_SIZE  = 1024
MASK_SIZE   = 512
BATCH_SIZE  = 4    # 1024×1024 images are large
NUM_WRITERS = 8

# SAM normalization constants
PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
PIXEL_STD  = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)


def preprocess(imgs_pil: list, device: torch.device) -> torch.Tensor:
    tensors = []
    for img in imgs_pil:
        arr = np.array(img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR), dtype=np.float32)
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        tensors.append(t)
    x = torch.cat(tensors, dim=0).to(device)
    x = (x - PIXEL_MEAN.to(device)) / PIXEL_STD.to(device)
    return x


def load_pair(img_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray, Image.Image]:
    with Image.open(img_path) as img:
        img_pil = img.convert("RGB")
    with Image.open(mask_path) as msk:
        msk_r = msk.convert("L").resize((MASK_SIZE, MASK_SIZE), Image.NEAREST)
    img_raw = np.array(img_pil.convert("L").resize((MASK_SIZE, MASK_SIZE), Image.BILINEAR), dtype=np.float32)
    mask    = (np.array(msk_r, dtype=np.float32) > 0).astype(np.float32)
    return img_raw, mask, img_pil


def save_npz(out_path: Path, img_raw: np.ndarray, mask: np.ndarray, tokens: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(str(tmp), slice=img_raw, mask=mask, patch_tokens=tokens)
    tmp.rename(out_path)


def collect_pairs(images_dir: Path, labels_dir: Path, overwrite: bool, out_dir: Path):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_pairs  = collect_pairs(IMAGES_TR, LABELS_TR, args.overwrite, OUT_DIR)
    all_pairs += collect_pairs(IMAGES_TS, LABELS_TS, args.overwrite, OUT_DIR)

    total = len(list(IMAGES_TR.glob("*.png"))) + len(list(IMAGES_TS.glob("*.png")))
    print(f"Total images : {total}")
    print(f"À traiter    : {len(all_pairs)}  (déjà cachés : {total - len(all_pairs)})")
    if not all_pairs:
        print("Rien à faire.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device       : {device}")

    sys.path.insert(0, str(MRICORE_REPO))
    import argparse as _ap
    mri_args = _ap.Namespace(
        if_encoder_adapter=False, if_mask_decoder_adapter=False,
        if_encoder_lora_layer=False, if_decoder_lora_layer=False,
        if_update_encoder=False, num_cls=1, image_size=IMAGE_SIZE,
        encoder_adapter_depths=[0, 1, 10, 11], decoder_adapt_depth=2,
        encoder_lora_layer=[], normalize_type="sam",
    )
    from models.sam import sam_model_registry
    model = sam_model_registry["vit_b"](
        mri_args,
        checkpoint=str(MRICORE_CKPT),
        num_classes=1,
        image_size=IMAGE_SIZE,
        pretrained_sam=False,
    )
    model.to(device).eval()
    print(f"MRICore ViT-B chargé — sortie : ({BATCH_SIZE}, 256, 64, 64) → ({BATCH_SIZE}, 4096, 256)")

    errors = []
    prev_futures = []

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
                loaded   = [load_pair(ip, mp) for ip, mp, _ in batch]
                imgs_raw = [x[0] for x in loaded]
                masks    = [x[1] for x in loaded]
                imgs_pil = [x[2] for x in loaded]

                pixel_values = preprocess(imgs_pil, device)

                with torch.no_grad():
                    feat = model.image_encoder(pixel_values)  # (B, 256, 64, 64)
                    B, D, H, W = feat.shape
                    tokens = feat.permute(0, 2, 3, 1).reshape(B, H * W, D)  # (B, 4096, 256)
                    tokens_np = tokens.cpu().float().numpy()

                _wait(prev_futures)
                prev_futures = [
                    pool.submit(save_npz, out_path, imgs_raw[i], masks[i], tokens_np[i])
                    for i, (_, _, out_path) in enumerate(batch)
                ]
                pbar.update(len(batch))

        _wait(prev_futures)

    if errors:
        print(f"\n[WARN] {len(errors)} erreurs :")
        for e in errors[:5]:
            print(f"  {e}")
    print(f"\nNPZ sauvegardés dans : {OUT_DIR}")


if __name__ == "__main__":
    main()
