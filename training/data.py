from typing import Tuple, List, Optional

import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import nibabel as nib
except Exception:
    nib = None


class RandomVolumeDataset(Dataset):
    """Simple placeholder dataset that returns random volumes for quick smoke tests."""

    def __init__(self, length: int = 100, img_size: Tuple[int, int, int] = (32, 32, 32), in_channels=1):
        self.length = length
        self.img_size = img_size
        self.in_channels = in_channels

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = torch.randn(self.in_channels, *self.img_size)
        return x



def build_dataloaders(img_size,batch_size,splits=(0.8, 0.1, 0.1),total_samples: int = 250,num_workers: int = 2):
    t, v, te = splits

    n_train = int(total_samples * t)
    n_val = int(total_samples * v)
    n_test = total_samples - n_train - n_val

    train_ds = RandomVolumeDataset(length=n_train, img_size=img_size)
    val_ds = RandomVolumeDataset(length=n_val, img_size=img_size)
    test_ds = RandomVolumeDataset(length=n_test, img_size=img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


class MultiFolderDataset(Dataset):
    """Aggregate volumes from multiple folders into a single Dataset.

    Usage:
        ds = MultiFolderDataset(["/path/to/data1", "/path/to/data2"], img_size=(64,64,64))

    Supported file types: .npy, .pt/.pth (torch tensors), .nii/.nii.gz (if nibabel installed).
    Returns tensors of shape (C, D, H, W).
    """

    SUPPORTED_PATTERNS = ['*.nii', '*.nii.gz', '*.npy', '*.pt', '*.pth']

    def __init__(self, folders: List[str], img_size: Tuple[int, int, int] = (32, 32, 32), in_channels: int = 1, transform=None):
        self.folders = folders
        self.img_size = img_size
        self.in_channels = in_channels
        self.transform = transform

        self.files: List[str] = []
        for folder in folders:
            for pattern in self.SUPPORTED_PATTERNS:
                self.files.extend(sorted(glob.glob(os.path.join(folder, pattern))))

        if len(self.files) == 0:
            raise RuntimeError(f"No files found in the provided folders: {folders}")

    def __len__(self):
        return len(self.files)

    def _load(self, path: str) -> torch.Tensor:
        p = path.lower()
        if p.endswith('.nii') or p.endswith('.nii.gz'):
            if nib is None:
                raise RuntimeError('nibabel is required to load NIfTI files (pip install nibabel)')
            img = nib.load(path)
            arr = img.get_fdata(dtype=np.float32)
            tensor = torch.from_numpy(arr).float()
        elif p.endswith('.npy'):
            arr = np.load(path).astype(np.float32)
            tensor = torch.from_numpy(arr).float()
        elif p.endswith('.pt') or p.endswith('.pth'):
            obj = torch.load(path)
            if isinstance(obj, torch.Tensor):
                tensor = obj.float()
            elif isinstance(obj, np.ndarray):
                tensor = torch.from_numpy(obj).float()
            else:
                raise RuntimeError(f'Unsupported object in {path}: {type(obj)}')
        else:
            raise RuntimeError(f'Unsupported file type for {path}')

        # ensure shape (C, D, H, W)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 4:
            pass
        else:
            raise RuntimeError(f'Loaded tensor has unsupported shape {tensor.shape} from {path}')

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor

    def __getitem__(self, idx):
        path = self.files[idx]
        return self._load(path)

