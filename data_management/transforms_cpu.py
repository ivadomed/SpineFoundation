from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, EnsureTyped
import torch 
import numpy as np
from monai.transforms import MapTransform
from monai.data import MetaTensor

class ComputeSpacingDHWd(MapTransform):
    """
    Ajoute dans meta["spacing_dhw"] les espacements alignés avec img.shape[-3:].
    (On dérive les voxel sizes à partir de l'affine actuelle, après Orientationd.)
    """
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            if k in d and isinstance(d[k], MetaTensor):
                mt = d[k]

                # Affine actuelle (4x4), déjà en RAS après Orientationd
                A = np.asarray(mt.affine, dtype=float)   # shape (4,4)

                # Espacements le long des 3 axes de l'image
                # -> norme des colonnes de A[:3, :3]
                spacing_ijk = np.sqrt((A[:3, :3] ** 2).sum(0))  # (s0, s1, s2)

                mt.meta["spacing_dhw"] = spacing_ijk
        return d


def get_transforms_cpu():
    keys = ["image","label"]
    return Compose([
        LoadImaged(keys=keys, allow_missing_keys=True),
        EnsureChannelFirstd(keys=keys, allow_missing_keys=True),
        Orientationd(keys=keys, axcodes="RAS",labels=(('L', 'R'), ('P', 'A'), ('I', 'S')),allow_missing_keys=True),
        EnsureTyped(keys=keys, dtype=torch.float32, track_meta=True, allow_missing_keys=True),
        ComputeSpacingDHWd(keys=["image"]),
    ])
