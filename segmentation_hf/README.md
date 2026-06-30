# segmentation_hf

Entraînement d'une tête de segmentation binaire sur un backbone DINOv2 figé.
Support du cache de patch tokens pour un entraînement sans appel au backbone (fast path).

---

## Pipeline

```
slice_extraction/           ← slices 2D (PNG ou NPZ) depuis NIfTI BIDS
        ↓
cache_patch_tokens.py       ← GPU : backbone → patch_tokens dans chaque NPZ
        ↓
train_seg_from_hf.py        ← entraînement tête de segmentation
```

Deux chemins d'entraînement coexistent :

| Chemin | Données | Backbone appelé | Vitesse |
|--------|---------|-----------------|---------|
| **Fast path** (NPZ + tokens cachés) | `--npz_train_dir` | jamais | ~10-50× plus rapide |
| **Slow path** (NPZ sans tokens) | `--npz_train_dir` | à chaque batch | référence |
| **Image path** (PNG images+masks) | `--train_images/masks` | à chaque batch | original |

---

## Étape 0 — Extraire les slices (si pas déjà fait)

```bash
python slice_extraction/05_run_pipeline.py \
    --input-images /data/images \
    --input-labels /data/labels \
    --label-suffix _seg \
    --work-root /data/work \
    --no-tiling
```

Produit `work-root/02_final/image/{train,val}/` et `work-root/02_final/label/{train,val}/`.

---

## Étape 1 — Cacher les patch tokens (fast path)

```bash
python -m segmentation_hf.cache_patch_tokens \
    --data_dir /data/npz/train \
    --model_name /path/to/backbone \
    --processor_name /path/to/curia_snapshot \
    --suffix custom \
    --batch_size 64
```

La clé `patch_tokens_custom` est ajoutée dans chaque NPZ. Les fichiers déjà traités sont sautés automatiquement.

Pour plusieurs splits :

```bash
for split in train val; do
  python -m segmentation_hf.cache_patch_tokens \
      --data_dir /data/npz/$split \
      --model_name /path/to/backbone \
      --suffix custom
done
```

---

## Étape 2 — Entraîner la tête de segmentation

**Fast path (tokens cachés) :**

```bash
python -m segmentation_hf.train_seg_from_hf \
    --model_dir /path/to/backbone \
    --npz_train_dir /data/npz/train \
    --npz_val_dir   /data/npz/val \
    --patch_token_key patch_tokens_custom \
    --output_dir outputs_seg/run01 \
    --epochs 50 \
    --lr 1e-4 \
    --amp
```

**Image path (PNG, backbone appelé à chaque batch) :**

```bash
python -m segmentation_hf.train_seg_from_hf \
    --model_dir /path/to/backbone \
    --train_images /data/work/02_final/image/train \
    --train_masks  /data/work/02_final/label/train \
    --val_images   /data/work/02_final/image/val \
    --val_masks    /data/work/02_final/label/val \
    --output_dir outputs_seg/run01 \
    --image_size 224 \
    --tile_threshold 512 \
    --epochs 50 \
    --amp
```

**Avec W&B :**

```bash
python -m segmentation_hf.train_seg_from_hf \
    --model_dir /path/to/backbone \
    --npz_train_dir /data/npz/train \
    --npz_val_dir   /data/npz/val \
    --output_dir outputs_seg/run01 \
    --wandb \
    --wandb_project spine-seg \
    --wandb_run_name run01
```

---

## Structure des sorties

```
outputs_seg/run01/
├── best.pt          ← checkpoint meilleur val_dice
├── last.pt          ← checkpoint dernière époque
└── history.csv      ← epoch, train_loss, train_dice, val_loss, val_dice
```

---

## Format des fichiers NPZ

```
fichier.npz
├── "slice"          float32 (H, W)   — intensité brute de la slice IRM
├── "mask"           uint8   (H, W)   — masque de segmentation binaire
└── "patch_tokens"   float32 (N, D)   — tokens cachés (après cache_patch_tokens.py)
                                         N = (image_size / patch_size)²
                                         D = hidden_size du backbone
```

---

## Paramètres principaux

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `--model_dir` | — | Répertoire checkpoint backbone (format HuggingFace) |
| `--patch_token_key` | `patch_tokens` | Clé NPZ des tokens cachés |
| `--image_size` | 224 | Taille d'entrée backbone |
| `--tile_size` / `--tile_overlap_pct` / `--tile_threshold` | — | Paramètres de tiling |
| `--bce_weight` / `--dice_weight` | 0.5 / 0.5 | Pondération BCE + Dice |
| `--amp` | false | Mixed precision (recommandé) |
| `--epochs` | 50 | Nombre d'époques |
| `--lr` | 1e-4 | Learning rate |

---

## Relation avec `classification_hf`

| | `classification_hf` | `segmentation_hf` |
|---|---|---|
| Tâche | Classification multi-classe | Segmentation binaire |
| Cache tokens | `patch_tokens` (N, D) → pooling → (D,) | `patch_tokens` (N, D) → reshape → carte spatiale |
| Layout données | `class_0/…/img.npz` (sous-dossiers) | `img.npz` (répertoire plat) |
| Évaluation | AUC / accuracy, bootstrap IC 95% | Dice score par époque |
