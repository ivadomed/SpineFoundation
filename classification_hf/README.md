# classification_hf

Pipeline d'entraînement et d'évaluation de modèles de classification vertébrale, basé sur HuggingFace Transformers et un backbone DINOv2.

## Architecture générale

Le pipeline suit ce flux :
1. (Optionnel) Pré-calcul et cache des patch tokens du backbone → `cache_patch_tokens.py`
2. (Optionnel) Pré-calcul des features poolées à différents rayons → `cache_pooled_features.py`
3. Entraînement du classificateur → `train.py` + `trainer.py`
4. Évaluation bootstrap sur plusieurs runs → `bootstrap_eval.py`

---

## Fichiers Python

### `model.py`
Définit l'architecture du modèle de classification :
- `SelfAttentionPooling` — pooling par auto-attention sur les patch tokens
- `CrossAttentionPooling` — pooling par cross-attention guidé par le masque
- `Classifier` — tête de classification finale (backbone + pooling + MLP)

Architecture reprise verbatim depuis curia/raidium.

### `dataset.py`
Dataset PyTorch qui charge les patches 2D depuis un répertoire local (structure `split/label/patient_id/*.npz`).

Fonctionnalités :
- Chargement des patch tokens pré-cachés (NPZ) ou calcul à la volée
- Redimensionnement et dilation du masque de segmentation
- Masked average pooling des patch tokens selon le masque

Miroir local du pipeline de curia, sans dépendance HuggingFace Hub.

### `trainer.py`
Boucle d'entraînement basée sur `transformers.Trainer` :
- `compute_classification_metrics()` — calcule accuracy, AUC OvR macro et weighted
- `_merge_into_train_csv()` — écriture thread-safe (fcntl.flock) des résultats dans un CSV partagé
- `ClassificationTrainer` — sous-classe de Trainer avec loss cross-entropie et logging enrichi

### `train.py`
Point d'entrée unique : charge la config OmegaConf (YAML) et lance `ClassificationTrainer`.

### `bootstrap_eval.py`
Évaluation statistique par bootstrap — deux modes via `--mode` :

- **trained** (défaut) : agrège les logits de N runs d'entraînement (`val_predictions.npz` dans les sous-dossiers de `--runs-dir`), rééchantillonne pour estimer les IC à 95%
- **pretrained** : charge un modèle HuggingFace figé (`--model-name`, `--subfolder`), fait l'inférence sur des patches NPZ pré-cachés, puis bootstrap

Métriques communes : accuracy, AUC OvR macro, AUC OvR weighted.

Exemples :
```bash
# Mode trained
python -m classification_hf.bootstrap_eval --task nfn --runs-dir outputs_cls/rsna_nfn

# Mode pretrained (raidium/curia, avec dilation)
python -m classification_hf.bootstrap_eval --mode pretrained --task nfn --dilation-radius 4
```

### `cache_patch_tokens.py`
Pré-calcul et mise en cache des patch tokens du backbone DINOv2 pour tous les patches du dataset.
- Sauvegarde en NPZ (une clé par patch dans un fichier patient)
- I/O asynchrone avec threads dédiés lecture/écriture pour maximiser le débit disque
- Reprise possible : saute les patches déjà cachés

### `cache_pooled_features.py`
Pré-calcul des features poolées (après masked avg pooling) pour une grille de rayons de dilation.
- Utilisé pour l'étude d'ablation sur le rayon de dilation du masque
- Sauvegarde un vecteur par (patch, rayon) en NPZ

### `plot_dilation_study.py`
Visualisation des résultats de l'étude d'ablation sur le rayon de dilation :
- Charge les CSV de résultats par rayon
- Trace accuracy et AUC en fonction du rayon

---

## Configs YAML

| Fichier | Description |
|---|---|
| `config_foraminal.yaml` | Classification sténose foraminale |
| `config_subarticular.yaml` | Classification sténose sous-articulaire |
| `config_central.yaml` | Classification sténose centrale |
| `config_foraminal_dilation.yaml` | Étude dilation — sténose foraminale |
| `config_subarticular_dilation.yaml` | Étude dilation — sténose sous-articulaire |
| `config_central_dilation.yaml` | Étude dilation — sténose centrale |

---

## Scripts Shell

| Fichier | Description |
|---|---|
| `train.sh` | Lance un entraînement avec une config donnée |
| `run_dilation_study.sh` | Lance l'étude d'ablation sur plusieurs rayons de dilation |
| `run_bootstrap_study.sh` | Lance N runs indépendants puis évaluation bootstrap |

---

## Relation avec RSNA_downstream

`RSNA_downstream/` contient uniquement le pipeline d'extraction :
- `RSNAextractor.py` — extraction des patches 2D depuis les volumes NIFTI RSNA

Tout ce qui est inférence, évaluation, ou entraînement de modèle vit dans `classification_hf/`. Pour évaluer le modèle pré-entraîné raidium/curia, utiliser `bootstrap_eval.py --mode pretrained`.

### Dette technique intra classification_hf

Ces utilitaires sont dupliqués en interne et pourraient être centralisés dans un `utils.py` :
- `resize_mask()` / `_make_mask_transform()` — présent dans `dataset.py`, `cache_pooled_features.py`, `bootstrap_eval.py`
- `_masked_avg_pool()` — présent dans `dataset.py`, `cache_pooled_features.py`, `bootstrap_eval.py`
