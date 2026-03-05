# RSNA_downstream

Pipeline d'extraction de patches 2D depuis le dataset RSNA Lumbar Spine 2024 (format BIDS/NIfTI).

---

## Fichier

### `RSNAextractor.py`

Extrait des patches 2D annotés depuis des volumes NIfTI 3D et les sauvegarde en NPZ, prêts à être utilisés par `classification_hf/`.

**Flux :**
1. Parcourt les volumes NIfTI (T1w ou T2w selon la tâche)
2. Pour chaque volume, trouve le label NIfTI correspondant + son sidecar JSON (sévérité)
3. Sélectionne la coupe 2D contenant le plus de voxels positifs
4. (Optionnel) Recadre un patch centré sur l'annotation
5. Sauvegarde `slice` (image) + `mask` (annotation) dans un fichier NPZ, rangé dans un sous-dossier nommé par classe (`0/`, `1/`, `2/`)

**Structure de sortie :**
```
<out_dir>/
  0/   # Normal/Mild
  1/   # Moderate
  2/   # Severe
```

**Tâches supportées :**

| Tâche | Modalité | Axe de coupe | Pathologie |
|-------|----------|--------------|------------|
| `nfn` | T1w sag. | Sagittal (axe 0) | Neural Foraminal Narrowing |
| `ss`  | T2w ax.  | Axial (axe 2)    | Subarticular Stenosis |
| `scs` | T2w sag. | Sagittal (axe 0) | Spinal Canal Stenosis |

**Usage :**
```bash
python RSNA_downstream/RSNAextractor.py \
    --root /path/to/lumbar-rsna-challenge-2024 \
    --out-dir /path/to/data/patches_RSNA_raw_with_mask_nfn \
    --task nfn \
    --crop-size 200
```

---

## Étapes suivantes

Une fois les patches extraits, tout le reste (cache de features, entraînement, inférence, évaluation bootstrap) se fait dans `classification_hf/` :

```bash
# 1. Cacher les patch tokens du backbone
python -m classification_hf.cache_patch_tokens --data-dir /path/to/patches_nfn ...

# 2. Évaluer le modèle pré-entraîné raidium/curia
python -m classification_hf.bootstrap_eval --mode pretrained --task nfn

# 3. Ou entraîner un nouveau modèle
python -m classification_hf.train --config classification_hf/config_foraminal.yaml
```
