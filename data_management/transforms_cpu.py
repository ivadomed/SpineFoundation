from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, EnsureTyped
import torch 

def get_transforms_cpu():
    keys = ["image","label"]
    return Compose([
        LoadImaged(keys=keys, allow_missing_keys=True),
        EnsureChannelFirstd(keys=keys, allow_missing_keys=True),
        Orientationd(keys=keys, axcodes="RAS",labels=None ,allow_missing_keys=True),
        EnsureTyped(keys=keys, dtype=torch.float32, track_meta=True, allow_missing_keys=True),
    ])
