"""
This script contains data loading utilities.

Author: Thomas Dagonneau & Julien Laborde-Peyré
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from pathlib import Path
from monai.data import Dataset as MonaiDataset
import json

import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system") # To have more workers than default limit (which is 4 on many systems)


from .json_data_creator import create_data_manifest
from mae_training.transforms import get_transforms

def makemonaidataset(data_list, augment=False):

    transforms = get_transforms(augment=augment)
    return MonaiDataset(data=data_list, transform=transforms)


def build_dataloaders(splits=(0.8, 0.1, 0.1),shuffle_seed=None,data_path=False,json_path=False,json_save=False,rank=0):
    
    try: 
        json_path = os.path.abspath(Path(json_path))
        with open(json_path, 'r') as f:
            data_manifest = json.load(f)
        if rank==0:
            print(f"JSON manifest found at {json_path}.")
    except:
        if data_path==False:
            raise FileNotFoundError(f"JSON manifest not found and no data_path provided to create one.")
        if rank==0:
            print("Splits manifest doesn't exist, creating one.")
        data_manifest = create_data_manifest(data_path, splits, shuffle_seed, json_path, rank=rank)
        if json_save:
            with open(os.path.abspath(json_path), 'w') as f:
                    json.dump(data_manifest, f, indent=4)

    
    train_data = data_manifest.get("TRAINING", [])
    val_data = data_manifest.get("VALIDATION", [])
    test_data = data_manifest.get("TEST", [])
    
    if not train_data and not val_data and not test_data:
         raise RuntimeError(f"JSON file at {json_path} contains no data in TRAINING, VALIDATION, or TEST splits.")


    print("\n")

    train_ds = makemonaidataset(train_data, augment=False)

    val_ds = makemonaidataset(val_data, augment=False)
    test_ds = makemonaidataset(test_data, augment=False)

    # 3. Create DataLoaders
    #train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,pin_memory=True,persistent_workers=False,prefetch_factor=1,collate_fn=list_collate)
    #val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,pin_memory=True,persistent_workers=False,prefetch_factor=1,collate_fn=list_collate)
    #test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,pin_memory=True,persistent_workers=False,prefetch_factor=2,collate_fn=list_collate)
    return train_ds, val_ds, test_ds

def build_dataloaders_inference(json_path=False,only_validation=True,rank=0):
    
    if only_validation:
        try:
            json_path = os.path.abspath(Path(json_path))
            with open(json_path, 'r') as f:
                data_manifest = json.load(f)
            data=data_manifest.get("VALIDATION", [])
            val_ds = makemonaidataset(data, augment=False)
            return val_ds
        except:
            raise RuntimeError(f"JSON manifest not found for inference at {json_path}.")

    else:
        raise RuntimeError(f"Other splits than VALIDATION not implemented yet.")
    
    
    # 3. Create DataLoaders
    #train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,pin_memory=True,persistent_workers=False,prefetch_factor=1,collate_fn=list_collate)
    #val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,pin_memory=True,persistent_workers=False,prefetch_factor=1,collate_fn=list_collate)
    #test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,pin_memory=True,persistent_workers=False,prefetch_factor=2,collate_fn=list_collate)
    return val_ds

def list_collate(batch):
    return batch

