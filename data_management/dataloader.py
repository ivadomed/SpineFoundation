"""
This script contains data loading utilities.

Author: Thomas Dagonneau & Julien Laborde-Peyré
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset

from monai.data import Dataset as MonaiDataset

from training.transforms import get_transforms



def makemonaidataset(folder, files, img_size, img_resolution, augment):
    data_list = []
    for p in files:
        label_path = get_mask(folder,p)
        entry = {'image': p}
        entry['label'] = label_path
        data_list.append(entry)

    transforms = get_transforms(img_size, img_resolution, augment=augment)
    return MonaiDataset(data=data_list, transform=transforms)


def build_dataloaders(folders,img_size, img_resolution, batch_size,splits=(0.8, 0.1, 0.1),num_workers=2,shuffle_seed=None):

    t, v, te = splits
    sub_datasets = []


    for folder in folders:
        pattern = os.path.join(folder, "sub-*", "**", "anat", "*.nii.gz")
        found = sorted(glob.glob(os.path.join(pattern), recursive=True))
        found = [f for f in found if "ax" not in f.lower() and "cor" not in f.lower() and "preproc" not in f.lower()]
        sub_datasets.append(makemonaidataset(folder,found, img_size=img_size, img_resolution=img_resolution, augment=True))

    monai_ds=ConcatDataset(sub_datasets)
    total = len(monai_ds)
    if total == 0:
        raise RuntimeError('No files found')

    indices = list(range(total))
    if shuffle_seed is not None:
        rng = np.random.RandomState(shuffle_seed)
        rng.shuffle(indices)

    print(f"Total files found: {total}. Splitting into train/val/test with ratios {t}/{v}/{te}.")
    n_train = int(total * t)
    n_val = int(total * v)
    n_test = total - n_train - n_val
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    train_ds = Subset(monai_ds, train_indices)
    val_ds = Subset(monai_ds, val_indices)
    test_ds = Subset(monai_ds, test_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def get_mask(folder, img_file):


    labels_dir = os.path.join(folder, 'derivatives', 'labels')

    base = os.path.basename(img_file)
    base_noext = base
    for ext in ('.nii.gz', '.nii'):
        if base_noext.endswith(ext):
            base_noext = base_noext[:-len(ext)]
            break

    pattern = os.path.join(labels_dir, '**', f"{base_noext}*SC_seg*.nii*")
    matches = glob.glob(pattern, recursive=True)
    if matches:
        return matches[0]
    return None


