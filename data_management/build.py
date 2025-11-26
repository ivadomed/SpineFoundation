"""
This script centralises data management.

Author: Thomas Dagonneau & Julien Laborde-Peyré
"""

from .dataloader import build_dataloaders
from pathlib import Path



JSON_SPLIT_PATH="data_splits.json"

def build_datasets(data_path,json_path,splits,img_size,img_resolution,batch_size,num_workers,shuffle_seed):
    print(json_path)

    if isinstance(json_path, (str, Path)): #On a un chemin JSON
        data_path = False
        json_save = False
    elif json_path==True:
        json_path=JSON_SPLIT_PATH
        json_save = True
    else:
        json_save = False

    train_loader, val_loader, test_loader=build_dataloaders(img_size, img_resolution, batch_size,splits=(0.8, 0.1, 0.1),num_workers=2,
    shuffle_seed=None,data_path=data_path,json_path=json_path, json_save=json_save)

    return train_loader, val_loader, test_loader