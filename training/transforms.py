from typing import Tuple

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ResizeWithPadOrCropd,
    ScaleIntensityd,
    RandFlipd,
    RandAffined,
)


def get_transforms(img_size,augment= True, prob_flip= 0.2, prob_affine= 0.2):

    base = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=img_size),
        ScaleIntensityd(keys=["image"]),
    ]

    if augment:
        base.extend([
            RandFlipd(keys=["image"], spatial_axis=0, prob=prob_flip),
            RandAffined(keys=["image"], rotate_range=0.1, prob=prob_affine),
        ])

    return Compose(base)
