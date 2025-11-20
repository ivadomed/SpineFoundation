"""
This script contains data loading utilities.

Author: Thomas Dagonneau & Julien Laborde-Peyré
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from monai.data import Dataset as MonaiDataset

from preprocessing.transforms import get_transforms


def makemonaidataset(files, img_size, augment):
    data_list = [{'image': p} for p in files]
    transforms = get_transforms(img_size, augment=augment)
    return MonaiDataset(data=data_list, transform=transforms)


def build_dataloaders(img_size,batch_size,folders,splits=(0.8, 0.1, 0.1),num_workers=2,shuffle_seed=None):

    t, v, te = splits
    vol_files = []
    

    for folder in folders:
        pattern = os.path.join(folder, "sub-*", "anat", "*.nii.gz")
        vol_files.extend(sorted(glob.glob(os.path.join(pattern), recursive=True)))
    
    total = len(vol_files)
    if total == 0:
        raise RuntimeError('No files found')

    indices = list(range(total))
    if shuffle_seed is not None:
        rng = np.random.RandomState(shuffle_seed)
        rng.shuffle(indices)


    n_train = int(total * t)
    n_val = int(total * v)
    n_test = total - n_train - n_val
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    monai_ds = makemonaidataset(vol_files, img_size=img_size, augment=True)


    train_ds = Subset(monai_ds, train_indices)
    val_ds = Subset(monai_ds, val_indices)
    test_ds = Subset(monai_ds, test_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader




