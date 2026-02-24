# Slice Extraction Pipeline (ultra condensé)

## Utilisation

Commande complète (pipeline 01→05) :

```bash
python 05_run_pipeline.py \
  --input-images /path/images \
  --work-root /path/work \
  [--input-labels /path/labels] \
  [--tiling | --no-tiling] \
  [--tile-size 224] \
  [--tile-overlap-pct 25] \
  [--tile-threshold 512] \
  [--target-spacing 0.8] \
  [--interp bilinear] \
  [--train-ratio 0.9] \
  [--seed 42] \
  [--start-stage 1|2|3|4] \
  [--skip-existing | --no-skip-existing] \
  [--keep-intermediate]
```

## Flags principaux (`05_run_pipeline.py`)

- `--input-images` : dossier images (obligatoire)
- `--input-labels` : dossier labels (optionnel)
- `--work-root` : dossier de travail/sortie (obligatoire)
- `--tiling` / `--no-tiling` : activer/désactiver le tiling (stage 3)
- `--tile-size`, `--tile-overlap-pct`, `--tile-threshold` : paramètres du tiling
- `--target-spacing` : resampling in-plane (stage 2)
- `--interp` : `nearest|bilinear|bicubic|lanczos`
- `--start-stage` : reprise pipeline (`1,2,3,4`)
- `--skip-existing` / `--no-skip-existing` : skip ou recompute
- `--keep-intermediate` : conserve `01_extracted` et `02_resampled`

## Modes

- **Avec labels**: passer `--input-labels /path/labels`
- **Sans labels (image-only)**: ne pas passer `--input-labels`

## Stages (scripts)

- `01_extract_slices.py` : extraction 2D
  - flags clés: `--input-images --output-root [--input-labels]`
- `02_resample_inplane.py` : resample in-plane
  - flags clés: `--root --out-root [--with-labels|--no-labels]`
- `03_renumber_pairs.py` : renumber + tiling optionnel
  - flags clés: `--src-root --dst-root [--with-labels|--no-labels] [--tiling|--no-tiling]`
- `04_sanity_check_pairs.py` : sanity check
  - flags clés: `--root [--with-labels|--no-labels]`

## Exemples rapides

Avec labels + no tiling :

```bash
python 05_run_pipeline.py --input-images /data/img --input-labels /data/lbl --work-root /data/out --no-tiling
```

Sans labels + tiling :

```bash
python 05_run_pipeline.py --input-images /data/img --work-root /data/out --tiling
```
