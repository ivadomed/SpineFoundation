"""
This script contains data loading utilities.

Author: Thomas Dagonneau & Julien Laborde-Peyré
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from pathlib import Path
from monai.data import Dataset as MonaiDataset
import json

from training.transforms import get_transforms
from .json_data_creator import create_data_manifest


def makemonaidataset(data_list, img_size, img_resolution, augment):

    transforms = get_transforms(img_size, img_resolution,  augment=augment)
    return MonaiDataset(data=data_list, transform=transforms)


def build_dataloaders(img_size, img_resolution, batch_size,splits=(0.8, 0.1, 0.1),num_workers=2,shuffle_seed=None,data_path=False,json_path=False,json_save=False):
    
    try: #Si le JSON existe
        json_path = os.path.abspath(Path(json_path))
        with open(json_path, 'r') as f:
            data_manifest = json.load(f)
        print(f"JSON manifest found at {json_path}")
    except:
        print(json_path)
        if data_path==False:
            raise FileNotFoundError(f"JSON manifest not found and no data_path provided to create one.")
        print("Splits manifest doesn't exist, creating one...")
        data_manifest = create_data_manifest(data_path, splits, shuffle_seed, json_path)
        if json_save:
            with open(os.path.abspath(json_path), 'w') as f:
                    json.dump(data_manifest, f, indent=4)
            print(f"Data manifest successfully saved to {os.path.abspath(json_path)}")
    
    train_data = data_manifest.get("TRAINING", [])
    val_data = data_manifest.get("VALIDATION", [])
    test_data = data_manifest.get("TEST", [])
    
    if not train_data and not val_data and not test_data:
         raise RuntimeError(f"JSON file at {json_path} contains no data in TRAINING, VALIDATION, or TEST splits.")

    print(f"Data loaded (Image-Only): Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

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


