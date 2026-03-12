# Cleanup log — slices supprimées post-extraction

## 2026-03-10

### Critères de suppression
Fichiers PNG extraits dans `data_work/01_extracted/image/` supprimés manuellement
car ils correspondent à des séquences de calibration/technique sans valeur anatomique.

### Commandes exécutées

```bash
# 1. Field maps (B0 inhomogeneity maps)
find data_work/01_extracted/image -name "*fieldmap*" -delete
# Supprimés : 19,860 fichiers — principalement pd-mcgill

# 2. EPI sequences (_epi_ dans le nom)
find data_work/01_extracted/image -name "*_epi_*" -delete
# Supprimés : 18,650 fichiers — principalement pd-mcgill (sp1846x1846)

# 3. Single-band reference (avant séquences EPI multi-bande)
find data_work/01_extracted/image -name "*sbref*" -delete
# Supprimés : ~768 fichiers — ms-karolinska-2020

# 4. B0 mean (moyenne champ magnétique)
find data_work/01_extracted/image -name "*b0Mean*" -delete
# Supprimés : ~120 fichiers — sct-testing-large

# 5. Localizers (scouts basse résolution)
find data_work/01_extracted/image -name "*localizer*" -delete
# Supprimés : ~185 fichiers — ms-mayo-critical-lesions-2026
```

> Note : les commandes 3/4/5 ont été exécutées ensemble (`-o`), total = 2,474 fichiers.

### Total supprimé
| Type | Fichiers |
|---|---|
| fieldmap | 19,860 |
| _epi_ | 18,650 |
| sbref + b0Mean + localizer | 2,474 |
| **Total** | **40,984** |

### Datasets concernés
| Dataset | Type | Slices supprimées |
|---|---|---|
| pd-mcgill | fieldmap, epi | ~35,500 |
| ms-karolinska-2020 | sbref | ~768 |
| ms-mayo-critical-lesions-2026 | localizer | ~185 |
| sct-testing-large | b0Mean | ~120 |

### 2026-03-10 (suite)

```bash
# 6. Dossier vanderbilt-7t-swi (SWI, Chimap, phase/imag/real — aucune séquence anatomique)
rm -rf data_work/01_extracted/image/train/vanderbilt-7t-swi
rm -rf data_work/01_extracted/image/val/vanderbilt-7t-swi
# Supprimés : 1,774 fichiers

# 7. BOLD (fMRI) dans pd-mcgill
find data_work/01_extracted/image -name "*bold*" -delete
# Supprimés : 8,142 fichiers
```

| Type | Fichiers |
|---|---|
| vanderbilt-7t-swi (SWI/Chimap/phase) | 1,774 |
| bold (fMRI) | 8,142 |
| **Nouveau total supprimé** | **50,900** |

---

---

## 2026-03-11

### Datasets entiers exclus (hors scope colonne vertébrale)

```bash
rm -rf data_work/01_extracted/image/train/goldatlas
rm -rf data_work/01_extracted/image/val/goldatlas
# goldatlas : atlas/template anatomique — pas d'IRM patient — 3,713 fichiers

rm -rf data_work/01_extracted/image/train/levin-stroke
rm -rf data_work/01_extracted/image/val/levin-stroke
# levin-stroke : IRM cérébraux (AVC) — aucune colonne — 35,848 fichiers

rm -rf data_work/01_extracted/image/train/basel-mp2rage
rm -rf data_work/01_extracted/image/val/basel-mp2rage
# basel-mp2rage : cerveau uniquement (MP2RAGE) — aucune colonne
```

| Dataset | Raison | Slices supprimées |
|---|---|---|
| goldatlas | Atlas/template — pas d'IRM patient | 3,713 |
| levin-stroke | Cerveau (AVC) — hors scope | 35,848 |
| basel-mp2rage | Cerveau uniquement (MP2RAGE) | - |
| hc-calgary-preschool | Cerveau pédiatrique uniquement | 89,644 |
| ms-nmo-beijing | Cerveau uniquement (NMO) | 109,806 |
| msseg_challenge_2016 | Cerveau uniquement (lésions MS) | 17,884 |
| msseg_challenge_2021 | Cerveau uniquement (lésions MS) | 25,058 |
| nih-ms-mp2rage | Cerveau uniquement (MS MP2RAGE) | 170,972 |
| synthrad-challenge-2023 | Pelvis uniquement (MRI synthétique depuis CT) | 21,008 |
| uk-biobank | Cerveau uniquement | 910,865 |

### Datasets exclus du config uniquement (0 slices extraites)

| Dataset | Raison |
|---|---|
| eeg-epilepsy | Format EDF (EEG) — aucun NIfTI |

---

### Suppression partielle dans un dataset existant

```bash
# desc-registered dans head-neck-tumor-challenge-2024
# Images recalées (registration) — doublons artificiels
find data_work/01_extracted/image/train/head-neck-tumor-challenge-2024 \
     data_work/01_extracted/image/val/head-neck-tumor-challenge-2024 \
     -name "*registered*" -delete
# Supprimés : 13,530 fichiers
```

| Dataset | Pattern | Slices supprimées |
|---|---|---|
| head-neck-tumor-challenge-2024 | `*registered*` | 13,530 |
| inspired | `*MPM*`, `*DWI*`, `*_dir-AP_*`, `*_dir-PA_*` | 739,299 |
| lumbar-marseille | `*localizer*` | 30 |
| nih-ms-mp2rage, ms-rennes-mp2rage, marseille-3t-mp2rage, hc-leipzig-7t-mp2rage, ms-dresden-mp2rage-2025 | `*UNIT1*` (fond bruité — remplacé par inv-2) | 242,609 |
| nih-ms-mp2rage, ms-dresden-mp2rage-2025 | `*T1map*` (quantitatif — contraste non anatomique) | 97,802 |

---

### À faire dans is_ignored (01_extract_slices.py)
Ajouter ces patterns pour éviter de les extraire lors des prochaines runs :
```python
if any(kw in name for kw in ("fieldmap", "_epi_", "sbref", "b0mean", "localizer")):
    return True
```
