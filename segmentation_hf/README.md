# segmentation_hf

Pipeline d'entraînement de tête de segmentation sur un backbone DINOv2 figé, avec support du cache de patch tokens pour un entraînement rapide (backbone jamais appelé en train).

---

## Architecture générale

```
slice_extraction/         ← extraction des slices 2D depuis NIfTI (BIDS)
        ↓  NPZ (slice + mask)
segmentation_hf/cache_patch_tokens.py   ← GPU : backbone → patch_tokens dans NPZ
        ↓  NPZ (slice + mask + patch_tokens)
segmentation_hf/train_seg_from_hf.py    ← entraînement tête seg (fast path ou slow path)
```

Deux chemins d'entraînement coexistent :

| Chemin | Données | Backbone | Vitesse |
|--------|---------|----------|---------|
| **Fast path** (NPZ + tokens cachés) | `--npz_train_dir` | jamais appelé | ~10-50× plus rapide |
| **Slow path** (NPZ sans tokens) | `--npz_train_dir` | appelé à chaque batch | référence |
| **Image path** (PNG images+masks) | `--train_images/masks` | appelé à chaque batch | original |

---

## Fichiers Python

### `model.py`
- `PatchWiseSegHead` — tête de segmentation Conv2d → GELU → Conv2d (1 canal)
- `FrozenBackboneWithSegHead` — backbone DINOv2 figé + tête
  - `extract_patch_tokens()` — extrait les patch tokens (CLS ignoré), reshape en grille spatiale
  - `forward(x)` — chemin normal : backbone → seg_head → upsample
  - `forward_from_tokens(patch_tokens, target_hw)` — **fast path** : skip le backbone, reshape les tokens cachés → seg_head → upsample

### `dataset.py`
Deux datasets :

**`PairedSegmentationDataset`** (chemin image)
- Lit des fichiers PNG depuis `image_dir/` et `mask_dir/`
- Support du tiling pour les grandes images (`tile_threshold`, `tile_overlap_pct`)
- Normalisation z-score de l'image complète avant tiling

**`NpzSegmentationDataset`** (chemin NPZ)
- Lit des fichiers NPZ depuis un répertoire plat
- Fast path : si la clé `patch_token_key` est présente → retourne `(patch_tokens: N×D, mask: 1×S×S)`
- Slow path : retourne `(image: 1×S×S, mask: 1×S×S)` normalisée
- `load_raw_pair()` compatible avec l'évaluation full-image

### `trainer.py`
- `run_train_epoch()` — epoch standard (chemin image ou NPZ slow)
- `run_train_epoch_from_tokens()` — **fast path** : backbone jamais appelé
- `run_full_image_eval()` — évaluation tile-stitch pour le chemin image
- `run_full_image_eval_npz()` — évaluation NPZ (fast ou slow selon tokens présents)
- `predict_full_image_detiled()` — reconstruction full-image depuis les tiles (eval + overlays W&B)
- `capture_val_overlays()` — génère des overlays RGB pour W&B (GT=vert, pred=rouge, overlap=jaune)
- `train()` — point d'entrée, détecte automatiquement le chemin (NPZ ou image)

### `cache_patch_tokens.py`
Pré-calcul des patch tokens du backbone et mise en cache dans les NPZ.
- Répertoire plat (pas de sous-dossiers de classes, contrairement à `classification_hf`)
- Écriture atomique (`tempfile` + `os.replace`) — reprise sûre si interruption
- I/O asynchrone : 16 threads lecture, 8 threads écriture pendant que le GPU traite le batch suivant
- Idempotent : saute les fichiers déjà traités (contrôlable via `--overwrite`)

### `config.py`
Dataclass `TrainConfig` + parseur CLI complet. Champs principaux :

| Champ | Description |
|-------|-------------|
| `model_dir` | Répertoire du checkpoint backbone (format HuggingFace) |
| `train_images` / `train_masks` | Répertoires PNG images/masques (chemin image) |
| `val_images` / `val_masks` | Idem pour la validation |
| `npz_train_dir` / `npz_val_dir` | Répertoires NPZ (chemin fast path) |
| `patch_token_key` | Clé NPZ des tokens cachés (défaut : `patch_tokens`) |
| `image_size` | Taille d'entrée backbone (défaut : 224) |
| `tile_size` / `tile_overlap_pct` / `tile_threshold` | Paramètres de tiling |
| `bce_weight` / `dice_weight` | Pondération BCE + Dice |
| `amp` | Mixed precision |

### `losses.py`
- `dice_loss_with_logits()` — 1 − Dice
- `compute_dice_score()` — Dice binarisé (seuil 0.5)

---

## Usage

### Étape 0 — Extraire les slices depuis un dataset BIDS

```bash
python slice_extraction/05_run_pipeline.py \
    --repos https://github.com/org/bids-dataset \
    --input-labels /path/to/derivatives/labels \
    --label-suffix _seg \
    --work-root /data/work \
    --no-tiling
```

Cela produit `work-root/02_final/image/{train,val}/` et `work-root/02_final/label/{train,val}/`.

Pour le chemin NPZ, convertir ensuite les PNG en NPZ (script externe) ou utiliser directement les PNG avec le chemin image.

---

### Étape 1 — Cacher les patch tokens (fast path)

```bash
python -m segmentation_hf.cache_patch_tokens \
    --data_dir /data/npz/train \
    --model_name /path/to/backbone \
    --processor_name /path/to/curia_snapshot \
    --suffix custom \
    --batch_size 64
```

La clé `patch_tokens_custom` est ajoutée dans chaque NPZ. Les fichiers déjà traités sont sautés automatiquement.

Pour plusieurs répertoires (train + val) :
```bash
for split in train val; do
  python -m segmentation_hf.cache_patch_tokens \
      --data_dir /data/npz/$split \
      --model_name /path/to/backbone
done
```

---

### Étape 2 — Entraîner la tête de segmentation

**Fast path (tokens cachés, backbone jamais appelé) :**
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

**Chemin image (PNG, backbone appelé à chaque batch) :**
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

## Structure des répertoires de sortie

```
outputs_seg/run01/
├── best.pt          ← checkpoint meilleur val_dice
├── last.pt          ← checkpoint dernière époque
└── history.csv      ← epoch, train_loss, train_dice, val_loss, val_dice
```

---

## Structure des fichiers NPZ

```
fichier.npz
├── "slice"          float32 (H, W)   — intensité brute de la slice IRM
├── "mask"           uint8   (H, W)   — masque de segmentation binaire
└── "patch_tokens"   float32 (N, D)   — tokens cachés (après cache_patch_tokens.py)
                                         N = (image_size / patch_size)²
                                         D = hidden_size du backbone
```

---

## Format attendu du backbone

Le backbone doit être un répertoire HuggingFace-compatible contenant :
- `config.json` avec `hidden_size` (ou `embed_dim`) et `patch_size`
- `model.safetensors` (ou `pytorch_model.bin`)
- `preprocessor_config.json` (pour `AutoImageProcessor`)

Compatible avec le modèle `raidium/curia` et tout checkpoint DINOv2 fine-tuné.

---

## Relation avec `classification_hf`

| | `classification_hf` | `segmentation_hf` |
|---|---|---|
| Tâche | Classification multi-classe | Segmentation binaire |
| Sortie modèle | vecteur (D,) → logits (C,) | carte spatiale (H, W) |
| Pooling | masked avg pooling | aucun (spatial préservé) |
| Cache tokens | `patch_tokens` (N, D) | `patch_tokens` (N, D) |
| Post-cache | `cache_pooled_features.py` → `.pt` | direct : reshape → seg_head |
| Layout données | `class_0/…/img.npz` (sous-dossiers) | `img.npz` (répertoire plat) |
| Évaluation | bootstrap avec IC 95% | Dice score par époque |
