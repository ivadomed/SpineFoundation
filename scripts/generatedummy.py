import numpy as np
import nibabel as nib
from pathlib import Path

# 1. Créer un volume 3D (1×1×1) rempli de zéros
arr = np.zeros((1, 1, 1), dtype=np.float32)

# 2. Définir une affine (résolution / orientation)
sx, sy, sz = 1.0, 1.0, 1.0  # voxel size en mm
affine = np.array([
    [sx, 0, 0, 0],
    [0, sy, 0, 0],
    [0, 0, sz, 0],
    [0, 0, 0, 1],
], dtype=float)

# 3. Chemin de sortie
out_path = Path("./data_management/dummy/dummy_mask.nii.gz")
out_path.parent.mkdir(parents=True, exist_ok=True)

# 4. Créer et sauvegarder le NIfTI
nii = nib.Nifti1Image(arr, affine)
nib.save(nii, str(out_path))

print("Saved:", out_path.resolve())
print("NIfTI shape:", arr.shape)
