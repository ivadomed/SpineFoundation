from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, EnsureTyped, RandFlipd, RandRotated, SpatialPadd, RandSpatialCropd, RandLambdad, RandBiasFieldd, RandAffined, RandGaussianNoised, RandGaussianSharpend, ResizeWithPadOrCropd, RandScaleIntensityd, NormalizeIntensityd, ToTensord
import torch 
import numpy as np
from utils.augmentations import aug_sqrt, aug_sin, aug_exp, aug_sig, aug_laplace, aug_inverse, ComputeSpacingDHWd, GPUResampleAug3D


def get_transforms_cpu(img_size, target_res, augment = False):
    keys = ["image"]
    transforms = [
        LoadImaged(keys=keys, allow_missing_keys=True),
        EnsureChannelFirstd(keys=keys, allow_missing_keys=True),
        Orientationd(keys=keys, axcodes="RAS",labels=(('L', 'R'), ('P', 'A'), ('I', 'S')),allow_missing_keys=True),
        EnsureTyped(keys=keys, dtype=torch.float32, track_meta=True, allow_missing_keys=True),
        ComputeSpacingDHWd(keys=keys),
        GPUResampleAug3D(keys=keys, img_size=img_size, target_res=target_res)
    ]

    if augment:
        transforms += [
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandRotated(keys=keys, prob=0.5, range_y=0.1),
            RandLambdad(keys=keys,func=aug_sqrt,prob=0.05,),
            RandLambdad(keys=keys,func=aug_sin,prob=0.05,),
            RandLambdad(keys=keys,func=aug_exp,prob=0.05,),
            RandLambdad(keys=keys,func=aug_sig,prob=0.05, ),
            RandLambdad(keys=keys,func=aug_laplace,prob=0.05,),
            RandLambdad(keys=keys,func=aug_inverse,prob=0.05, ),   
            RandBiasFieldd(keys=keys,prob=0.05),
            RandAffined(keys=keys,prob=0.05, padding_mode="zeros", mode=["bilinear"]), 
            RandGaussianNoised(keys=keys, mean=0.0, std=0.1, prob=0.05),
            RandGaussianSharpend(keys=keys, prob=0.05),   
            ResizeWithPadOrCropd(keys=keys, spatial_size=(6, 100, 100)),
            RandScaleIntensityd(keys=keys, factors=(0.8, 1.2), prob=1), 
            NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),  
            ToTensord(keys=keys)
        ]


    return Compose(transforms)
