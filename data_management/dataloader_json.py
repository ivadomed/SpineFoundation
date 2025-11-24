"""
This script contains data loading utilities, modified for unsupervised learning (MAE).
It loads image paths from a JSON file, ignoring the need for labels/masks.

Author: Thomas Dagonneau & Julien Laborde-Peyré
"""
import os
import glob
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from typing import List, Dict, Tuple

from monai.data import Dataset as MonaiDataset

from training.transforms import get_transforms


def makemonaidataset(data_list: List[Dict[str, str]], img_size, img_resolution, augment):
    """
    Creates a MONAI Dataset from a pre-defined list of dictionaries.
    
    Since we are using MAE (unsupervised), the expected data_list format is: 
    [{'image': '/full/path/img.nii.gz'}, ...]
    """
    
    # We only need transforms for the image key
    transforms = get_transforms(img_size, img_resolution,  augment=augment)

    # Use the prepared list directly. MONAI will only load the keys present in the data_list.
    return MonaiDataset(data=data_list, transform=transforms)


def build_dataloaders_from_json(
    json_path: str, 
    img_size, 
    img_resolution, 
    batch_size: int, 
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds data loaders by loading pre-defined splits (TRAINING, VALIDATION, TEST) 
    from a JSON manifest file for unsupervised training (MAE).
    """
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON manifest not found at: {json_path}. Please run 'create_data_manifest.py' first.")

    with open(json_path, 'r') as f:
        data_manifest = json.load(f)

    # 1. Extract the pre-defined splits
    train_data = data_manifest.get("TRAINING", [])
    val_data = data_manifest.get("VALIDATION", [])
    test_data = data_manifest.get("TEST", [])
    
    if not train_data and not val_data and not test_data:
         raise RuntimeError(f"JSON file at {json_path} contains no data in TRAINING, VALIDATION, or TEST splits.")

    print(f"Data loaded from JSON (Image-Only): Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    # 2. Create Datasets using the pre-loaded data lists
    # Training usually requires heavy augmentation (augment=True) for MAE
    train_ds = makemonaidataset(train_data, img_size=img_size, img_resolution=img_resolution, augment=True)
    # Validation/Test typically use minimal or no augmentation (augment=False)
    val_ds = makemonaidataset(val_data, img_size=img_size, img_resolution=img_resolution, augment=False)
    test_ds = makemonaidataset(test_data, img_size=img_size, img_resolution=img_resolution, augment=False)

    # 3. Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


