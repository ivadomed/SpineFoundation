from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, EnsureTyped, RandFlipd, RandRotated, SpatialPadd, RandSpatialCropd, RandLambdad, RandBiasFieldd, RandAffined, RandGaussianNoised, RandGaussianSharpend, ResizeWithPadOrCropd, RandScaleIntensityd, NormalizeIntensityd
import torch 
import numpy as np
from .augment import aug_sqrt, aug_sin, aug_exp, aug_sig, aug_laplace, aug_inverse, ComputeSpacingDHWd, GPUResampleAug3D

from monai.transforms import MapTransform

class LoadImagedWithPathDebug(MapTransform):
    def __init__(self, keys, allow_missing_keys=True, **kwargs):
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.loader = LoadImaged(keys=keys, allow_missing_keys=allow_missing_keys, **kwargs)

    def __call__(self, data):
        d = dict(data)
        try:
            return self.loader(d)
        except Exception as e:
            for k in self.keys:
                if k in d:
                    print(f"[LOAD-CRASH] key={k} path={d[k]}", flush=True)
            raise


def get_transforms(augment=False):
    keys=["image","labels"]
    transforms=[
        LoadImagedWithPathDebug(keys=keys, allow_missing_keys=True),
        EnsureChannelFirstd(keys=keys, allow_missing_keys=True),
        Orientationd(keys=keys, axcodes="RAS",labels=(('L','R'),('P','A'),('I','S')),allow_missing_keys=True),
        EnsureTyped(keys=keys, dtype=torch.float32, track_meta=True, allow_missing_keys=True),
    ]

    if augment:
        transforms += [
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandRotated(keys=keys, prob=0.5, range_y=0.1),
            RandLambdad(keys=keys, func=aug_sqrt, prob=0.05),
            RandLambdad(keys=keys, func=aug_sin, prob=0.05),
            RandLambdad(keys=keys, func=aug_exp, prob=0.05),
            RandLambdad(keys=keys, func=aug_sig, prob=0.05),
            RandLambdad(keys=keys, func=aug_laplace, prob=0.05),
            RandLambdad(keys=keys, func=aug_inverse, prob=0.05),
            RandBiasFieldd(keys=keys, prob=0.05),
            RandAffined(keys=keys, prob=0.05, padding_mode="zeros", mode=["bilinear"]),
            RandGaussianNoised(keys=keys, mean=0.0, std=0.1, prob=0.05),
            RandGaussianSharpend(keys=keys, prob=0.05),
            ResizeWithPadOrCropd(keys=keys, spatial_size=(6,100,100)),
            RandScaleIntensityd(keys=keys, factors=(0.8,1.2), prob=1),
            NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),
        ]

    return Compose(transforms+[ComputeSpacingDHWd(keys=keys)])
