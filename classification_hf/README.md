# classification_hf

Classification de sévérité (Normal / Moderate / Severe) sur les données RSNA Lumbar Spine 2024.

Trois conditions : spinal canal stenosis (`scs`), neural foraminal narrowing (`nfn`), subarticular stenosis (`ss`).

---

## Pipeline

```
data_dir/{0,1,2}/*.npz
        ↓  cache_patch_tokens.py  (GPU)
patch_tokens_{suffix} ajouté dans chaque NPZ
        ↓  cache_pooled_features.py  (CPU)
~/.cache/classification_hf/pooled_features_*.pt
        ↓  train.py --config configs/rsna_*.yaml
Classifier entraîné → outputs_cls/
```

---

## Structure des données

```
data_dir/
├── 0/          ← Normal
│   └── sub-12345_acq-sag_....npz
├── 1/          ← Moderate
│   └── ...
└── 2/          ← Severe
    └── ...
```

Chaque NPZ doit contenir :

| Clé | Type | Description |
|-----|------|-------------|
| `slice` | float32 (H, W) | Intensité brute de la slice IRM |
| `mask` | uint8 (H, W) | Masque binaire de la région d'intérêt |
| `spacing_mm` | float | Espacement in-plane (requis si `crop_cm` activé) |

---

## Étape 1 — Cacher les patch tokens (GPU)

```bash
python -m classification_hf.cache_patch_tokens \
    --data_dir /path/to/RSNA_patches_scs \
    --model_name /path/to/backbone \
    --processor_name /path/to/curia_snapshot \
    --suffix curia_crop4cm \
    --crop_cm 4.0 \
    --batch_size 64
```

La clé `patch_tokens_curia_crop4cm` est ajoutée dans chaque NPZ. Les fichiers déjà traités sont sautés automatiquement (`--overwrite` pour forcer).

Pour plusieurs splits :

```bash
for split in train val; do
  python -m classification_hf.cache_patch_tokens \
      --data_dir /path/to/RSNA_patches_scs/$split \
      --model_name /path/to/backbone \
      --suffix curia_crop4cm \
      --crop_cm 4.0
done
```

---

## Étape 2 — Pooler les features (CPU)

```bash
python -m classification_hf.cache_pooled_features \
    --data_dir /path/to/RSNA_patches_scs \
    --token_key patch_tokens_curia_crop4cm \
    --cache_suffix curia_crop4cm
```

Produit `~/.cache/classification_hf/pooled_features_RSNA_patches_scs_curia_crop4cm.pt`.

Cette étape n'est à faire qu'une fois — `train.py` chargera le `.pt` automatiquement au démarrage.

---

## Étape 3 — Entraîner

```bash
python -m classification_hf.train --config classification_hf/configs/rsna_scs_crop4cm.yaml
```

### Configs disponibles

| Config | Tâche | Remarques |
|--------|-------|-----------|
| `rsna_scs_crop4cm.yaml` | SCS | crop 4 cm, features cachées |
| `rsna_nfn_crop4cm_resnet.yaml` | NFN | TokenGridClassifier (CNN spatial) |
| `rsna_ss_fold.yaml` | SS | fold split sujet-niveau |
| `rsna_*_fold_curia.yaml` | * | backbone Curia fine-tuné |
| `rsna_*_mricore.yaml` | * | backbone MRICore |
| `rsna_*_spine_only.yaml` | * | données spine uniquement |

Les champs principaux d'un fichier de config :

```yaml
model:
  model_name: /path/to/backbone
  num_classes: 3
  attention_cfg: null       # null = linear head simple

data_dir: /path/to/RSNA_patches_scs
fold_split_csv: /path/to/fold_split_RSNA.json
fold_column: regime_all_split_1_set

epochs: 50
batch_size: 512
learning_rate: 0.005
use_feature_caching: true
cache_suffix: curia_crop4cm
```

---

## Étape 4 — Évaluer sur le test set

```bash
# Un seul run
python -m classification_hf.eval_test \
    --pred outputs_cls/rsna_scs_crop4cm \
    --task scs

# Pooler plusieurs folds (stack des logits)
python -m classification_hf.eval_test \
    --pred outputs_cls/rsna_scs_fold/fold_1 outputs_cls/rsna_scs_fold/fold_2 \
    --task scs

# Toutes les tâches en une commande
python -m classification_hf.eval_test \
    --task nfn scs ss \
    --pred outputs_cls/rsna_nfn_fold outputs_cls/rsna_scs_fold outputs_cls/rsna_ss_fold
```

Métriques : cross-entropy, macro AUC (OvR), matrice de confusion, bootstrap IC 95%.

---

## Structure des sorties

```
outputs_cls/rsna_scs_crop4cm/
├── best.pt                  ← checkpoint meilleur val_loss
├── last.pt                  ← checkpoint dernière époque
├── test_predictions.npz     ← logits + labels (pour eval_test.py)
└── history.csv              ← epoch, train_loss, val_loss, val_acc
```

---

## Modèles

| Classe | Description |
|--------|-------------|
| `Classifier` | Masked avg pool + linear (avec support attention cross/self) |
| `TokenGridClassifier` | CNN résiduel sur la grille spatiale 2D des patch tokens |
| `MaskedBackboneClassifier` | Backbone DINOv2 non-figé + masked avg pool + linear |
