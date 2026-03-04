# RSNA 2024 Lumbar Spine — Downstream Evaluation

Pipeline en trois étapes pour évaluer le modèle pré-entraîné [`raidium/curia`](https://huggingface.co/raidium/curia)
sur le dataset BIDS `lumbar-rsna-challenge-2024` :

1. **`RSNAextractor.py`** — extrait des patches 2D (`.npz`) depuis les volumes NIfTI + labels
2. **`cache_features_to_npz.py`** — cache les features DINOv2 dans les `.npz` (optionnel, accélère l'entraînement)
3. **`eval_pretrained.py`** — évalue la tête de classification correspondante du modèle

---

## Tâches

| Acronyme | Pathologie | Modalité | Acquisition | Subfolder modèle |
|---|---|---|---|---|
| `nfn` | Neural Foraminal Narrowing | T1w | `acq-sag` | `neural_foraminal_narrowing` |
| `ss`  | Subarticular Stenosis      | T2w | `acq-ax`  | `subarticular_stenosis`      |
| `scs` | Spinal Canal Stenosis      | T2w | `acq-sag` | `spinal_canal_stenosis`      |

### Distribution des labels

#### nfn — Neural Foraminal Narrowing
- 19 339 labels total
  - Normal/Mild : 15 033 (classe 0)
  - Moderate    : 3 511  (classe 1)
  - Severe      : 765    (classe 2)
  - *(30 avec `nan` ignorés)*

#### ss — Subarticular Stenosis
- 18 830 labels total
  - Normal/Mild : 13 403 (classe 0)
  - Moderate    : 3 606  (classe 1)
  - Severe      : 1 816  (classe 2)
  - *(5 avec `nan` ignorés)*

#### scs — Spinal Canal Stenosis
- 9 560 labels total
  - Normal/Mild : 8 374 (classe 0)
  - Moderate    : 722   (classe 1)
  - Severe      : 464   (classe 2)

### Classes communes

| `PathologySeverity` (JSON sidecar) | Classe |
|---|---|
| Normal/Mild | 0 |
| Moderate    | 1 |
| Severe      | 2 |

---

## Étape 1 — Extraction des patches (`RSNAextractor.py`)

Lit les volumes NIfTI et leurs labels BIDS, sélectionne la coupe sagittale (ou axiale) la plus annotée,
et sauvegarde un fichier `.npz` par label contenant :
- `slice` : coupe 2D (`float32`)
- `mask`  : masque binaire 2D (`uint8`)

### Arguments

| Argument | Défaut | Description |
|---|---|---|
| `--task` | *requis* | `nfn`, `ss` ou `scs` |
| `--root` | `/home/.../lumbar-rsna-challenge-2024` | Racine du dataset BIDS |
| `--out-dir` | *requis* | Dossier de sortie (structure `0/`, `1/`, `2/`) |
| `--crop-size` | `0` (coupe entière) | Taille du crop centré sur l'annotation |

### Exemples

```bash
python RSNAextractor.py --task nfn --out-dir /data/patches_nfn
python RSNAextractor.py --task ss  --out-dir /data/patches_ss  --crop-size 200
python RSNAextractor.py --task scs --out-dir /data/patches_scs --crop-size 200
```

### Structure de sortie

```
out-dir/
    0/    *_label.npz    (Normal/Mild)
    1/    *_label.npz    (Moderate)
    2/    *_label.npz    (Severe)
```

---

## Étape 2 — Cache des features (`cache_features_to_npz.py`)

Passe le backbone DINOv2 de `raidium/curia` sur tous les `.npz` et ajoute une clé `features` de forme `(hidden_size,) float32`
via masked average pooling. Idempotent : les fichiers déjà traités sont ignorés.

### Arguments

| Argument | Défaut | Description |
|---|---|---|
| `--task` | *requis* | `nfn`, `ss` ou `scs` |
| `--model-name` | *requis* | Chemin local ou repo HF du snapshot curia |
| `--data-dir` | déduit de `--task` | Dossier contenant `0/`, `1/`, `2/` |
| `--batch-size` | `32` | Taille de batch |
| `--force` | `False` | Recalcule même si `features` existe déjà |

### Exemples

```bash
python cache_features_to_npz.py --task nfn --model-name raidium/curia
python cache_features_to_npz.py --task ss  --model-name /path/to/curia --batch-size 64
python cache_features_to_npz.py --task scs --model-name raidium/curia --force
```

---

## Étape 3 — Évaluation (`eval_pretrained.py`)

Charge la tête correspondante de `raidium/curia`, fait l'inférence sur les `.npz` extraits,
et produit un rapport complet (AUC OvR/OvO, accuracy, classification report, matrice de confusion).
Un fichier `.log` est sauvegardé automatiquement dans `--log-dir`.

### Arguments

| Argument | Défaut | Description |
|---|---|---|
| `--task` | *requis* | `nfn`, `ss` ou `scs` |
| `--data-dir` | déduit de `--task` | Dossier contenant `0/`, `1/`, `2/` |
| `--subfolder` | déduit de `--task` | Subfolder du modèle `raidium/curia` |
| `--batch-size` | `64` | Taille de batch pour l'inférence |
| `--log-dir` | `logs/` | Dossier parent pour les fichiers de log |

### Exemples

```bash
python eval_pretrained.py --task nfn
python eval_pretrained.py --task ss  --data-dir /data/patches_ss
python eval_pretrained.py --task scs --batch-size 32 --log-dir /data/logs
```

---

## Dépendances

```
nibabel
numpy
torch
transformers
scikit-learn
tqdm
```
