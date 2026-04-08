# Slice Extraction Pipeline

Extraction de slices 2D depuis des volumes NIfTI (BIDS ou autre), avec support git-annex et resample optionnel.

## Stages

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | — | Clone repos GitHub + `git annex get .` _(skip auto si pas de `--repos`)_ |
| 2 | `01_extract_slices.py` | Extraction 2D (axial/sagittal/isotropique) → PNG |
| 3 | `03_renumber_pairs.py` | Renumber + tiling optionnel |
| 4 | `04_sanity_check_pairs.py` | Vérification des paires image/label |
| 5 | `02_resample_inplane.py` | Resample in-plane _(optionnel, `--resample` requis)_ |

Dossiers intermédiaires :
```
work-root/
├── 00_cloned/      ← repos clonés (stage 1)
├── 01_extracted/   ← slices brutes (stage 2, nettoyé après stage 3)
├── 02_final/       ← sortie finale (stage 3)
└── 03_resampled/   ← si --resample activé (stage 5)
```

---

## Utilisation — pipeline complet

```bash
python slice_extraction/05_run_pipeline.py \
    --work-root /data/work \
    [OPTIONS]
```

### Avec un dataset BIDS sur GitHub (git-annex)

```bash
python slice_extraction/05_run_pipeline.py \
    --repos https://github.com/org/bids-dataset \
    --input-labels /path/to/derivatives/labels \
    --label-suffix _seg \
    --work-root /data/work \
    --no-tiling
```

`--input-images` est déduit automatiquement de `--repos` si omis.

### Avec des images locales

```bash
python slice_extraction/05_run_pipeline.py \
    --input-images /data/images \
    --input-labels /data/labels \
    --work-root /data/work \
    --label-suffix _seg \
    --tiling --tile-size 224 --tile-threshold 512
```

### Sans labels (image-only)

```bash
python slice_extraction/05_run_pipeline.py \
    --input-images /data/images \
    --work-root /data/work
```

### Avec resample in-plane (optionnel, dernier stage)

```bash
python slice_extraction/05_run_pipeline.py \
    --input-images /data/images \
    --work-root /data/work \
    --resample --target-spacing 0.8 --interp bilinear
```

### Reprise depuis un stage intermédiaire

```bash
python slice_extraction/05_run_pipeline.py \
    --work-root /data/work \
    --start-stage 3   # 1=clone, 2=extract, 3=renumber, 4=sanity, 5=resample
```

---

## Flags principaux (`05_run_pipeline.py`)

**Stage 1 — Clone**

| Flag | Défaut | Description |
|------|--------|-------------|
| `--repos URL [URL …]` | — | URLs de repos git à cloner |
| `--clone-root PATH` | `work-root/00_cloned` | Répertoire de destination |
| `--git-annex` / `--no-git-annex` | activé | Lancer `git annex get .` après clone |
| `--git-annex-jobs N` | 4 | Parallélisme pour git annex |

**Stage 2 — Extraction**

| Flag | Défaut | Description |
|------|--------|-------------|
| `--input-images PATH` | — | Racine des images NIfTI (obligatoire si pas de `--repos`) |
| `--input-labels PATH` | — | Racine des labels/derivatives NIfTI |
| `--label-suffix STR` | `_seg` | Suffixe label (`sub-01_T2w_seg.nii.gz`) |
| `--train-ratio FLOAT` | 0.9 | Proportion train/val |
| `--seed INT` | 42 | Graine pour la séparation train/val |
| `--clip-pct LO HI` | 0.5 99.5 | Percentiles de normalisation uint8 |
| `--iso-tol FLOAT` | 0.1 | Tolérance isotropie (ratio max/min spacing) |

**Stage 3 — Renumber + tiling**

| Flag | Défaut | Description |
|------|--------|-------------|
| `--tiling` / `--no-tiling` | activé | Tiling des grandes images |
| `--tile-size INT` | 224 | Taille de tuile (pixels) |
| `--tile-overlap-pct FLOAT` | 25.0 | Chevauchement (%) |
| `--tile-threshold INT` | 512 | Taille minimale déclenchant le tiling |

**Stage 5 — Resample (optionnel)**

| Flag | Défaut | Description |
|------|--------|-------------|
| `--resample` / `--no-resample` | désactivé | Activer le resample in-plane |
| `--target-spacing FLOAT` | 0.8 | Espacement cible isotrope (mm) |
| `--interp` | `bilinear` | `nearest \| bilinear \| bicubic \| lanczos` |

**Commun**

| Flag | Défaut | Description |
|------|--------|-------------|
| `--work-root PATH` | — | Racine des sorties (obligatoire) |
| `--start-stage {1..5}` | 1 | Reprendre depuis un stage |
| `--skip-existing` / `--no-skip-existing` | activé | Skip les fichiers déjà produits |
| `--keep-intermediate` | désactivé | Conserver `01_extracted/` |

---

## Utilisation via fichier de config (recommandé)

Tous les flags CLI peuvent être regroupés dans un fichier YAML et passés avec `--config` :

```bash
python slice_extraction/05_run_pipeline.py --config slice_extraction/config_pipeline_template.yaml
```

Le script `extract.sh` est un raccourci qui appelle cette commande directement.

### Fichiers de config disponibles

| Fichier | Description |
|---------|-------------|
| `config_pipeline_template.yaml` | Config complète : repos git-annex + images locales, image-only, `renumber: false` |
| `config_pipeline_template_2.yaml` | Variante avec une liste de `input_images` différente |
| `config_labels_only.yaml` | Re-clone les repos pour récupérer les `derivatives/` et extraire les labels sur des images déjà extraites (`skip_existing: true`) |

Les clés YAML correspondent exactement aux flags CLI (e.g. `--work-root` → `work_root:`).

---

## Scripts individuels

### `01_extract_slices.py`
```bash
python slice_extraction/01_extract_slices.py \
    --input-images /data/images \
    --output-root  /data/01_extracted \
    [--input-labels /data/labels] \
    [--label-suffix _seg]
```

### `02_resample_inplane.py`
```bash
python slice_extraction/02_resample_inplane.py \
    --root    /data/02_final \
    --target  0.8 \
    --out-root /data/03_resampled \
    [--with-labels | --no-labels]
```

### `03_renumber_pairs.py`
```bash
python slice_extraction/03_renumber_pairs.py \
    --src-root /data/01_extracted \
    --dst-root /data/02_final \
    [--tiling | --no-tiling] \
    [--with-labels | --no-labels]
```

### `04_sanity_check_pairs.py`
```bash
python slice_extraction/04_sanity_check_pairs.py \
    --root /data/02_final \
    [--with-labels | --no-labels]
```

---

## Structure de sortie

```
02_final/
├── image/
│   ├── train/
│   │   └── sub01__sub-01_T2w__sagittal__s0042__t000__sp0800x0800.png
│   └── val/
└── label/
    ├── train/
    │   └── sub01__sub-01_T2w__sagittal__s0042__t000__sp0800x0800.png
    └── val/
```

Nom de fichier : `{src}__{base}__{plane}__s{slice:04d}__t{tile:03d}__{sp}.png`
