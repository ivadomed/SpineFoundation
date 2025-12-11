"""
This script centralises data management.

Author: Thomas Dagonneau & Julien Laborde-Peyré
"""

from .dataloader import build_dataloaders
from .dataloader import build_dataloaders_inference
from pathlib import Path



JSON_SPLIT_PATH="./data_management/data_splits.json"

def build_datasets(data_path,json_path,splits,shuffle_seed,rank=0,inference=False,only_validation=True):
    if rank==0:
        print("\nDATA :\n")
    if isinstance(json_path, (str, Path)): #On a un chemin JSON
        data_path = False
        json_save = False

    elif json_path==True:
        json_path=JSON_SPLIT_PATH
        json_save = True
        
    else:
        json_save = False
    if inference:
         val_ds=build_dataloaders_inference(json_path=json_path,only_validation=True,rank=rank)
         return val_ds
    train_ds, val_ds, test_ds=build_dataloaders(splits=splits,
    shuffle_seed=shuffle_seed,data_path=data_path,json_path=json_path, json_save=json_save,rank=rank)

    return train_ds, val_ds, test_ds