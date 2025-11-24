from typing import Tuple

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ResizeWithPadOrCropd,
    ScaleIntensityd,
    RandFlipd,
    RandAffined,
    Spacingd,
    Orientationd
)


def get_transforms(img_size, resolution, augment= True, prob_flip= 0.2, prob_affine= 0.2):

    base = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RPI",labels=None),
        Spacingd(keys=["image", "label"], pixdim=resolution, mode=("bilinear")),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=img_size),
        ScaleIntensityd(keys=["image"]),
    ]

    if augment:
        base.extend([
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=prob_flip),
            RandAffined(keys=["image", "label"], rotate_range=0.1, prob=prob_affine),
        ])

    return Compose(base)

