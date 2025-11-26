import torch 
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ResizeWithPadOrCropd,
    ScaleIntensityd,
    RandFlipd,
    RandAffined,
    Spacingd,
    Orientationd,
    EnsureTyped,
    Lambdad,
)


def get_transforms(img_size,resolution,augment= True,prob_flip= 0.2,prob_affine= 0.2,):
    keys_img = ["image"]
    keys_both = ["image", "label"]

    base = [LoadImaged(keys=keys_both, allow_missing_keys=True),
            EnsureChannelFirstd(keys=keys_both, allow_missing_keys=True),
            Orientationd(keys=keys_both, axcodes="RPI",labels=None, allow_missing_keys=True),
            Spacingd(keys=keys_both,pixdim=resolution,mode=("bilinear", "nearest"),allow_missing_keys=True),
            ResizeWithPadOrCropd(keys=keys_both,spatial_size=img_size,allow_missing_keys=True),
            ScaleIntensityd(keys=keys_img),
            EnsureTyped(keys=keys_both,dtype=torch.float32,track_meta=False,allow_missing_keys=True),
    ]
    if augment:
        base.extend([RandFlipd(keys=keys_both,spatial_axis=0,prob=prob_flip,allow_missing_keys=True,),
                     RandAffined(keys=keys_both,rotate_range=0.1,prob=prob_affine,mode=("bilinear"),allow_missing_keys=True)])
    return Compose(base)
