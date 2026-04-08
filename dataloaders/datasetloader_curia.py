

import os
from pathlib import Path
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile, PngImagePlugin
import torch.distributed as dist
import random
from data.data_augmentation_custom import DataAugmentationDINO
from data.samplers import InfiniteSampler


class _FilteredImageFolder(ImageFolder):
    """ImageFolder that silently skips class subdirectories with no valid image files."""
    _VALID_EXTS = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}

    def find_classes(self, directory):
        classes, class_to_idx = super().find_classes(directory)
        non_empty = [
            c for c in classes
            if any(
                f.suffix.lower() in self._VALID_EXTS
                for f in Path(directory, c).iterdir()
                if f.is_file()
            )
        ]
        skipped = set(classes) - set(non_empty)
        if skipped:
            print(f"[datasetloader] skipping {len(skipped)} empty class(es): {sorted(skipped)}")
        return non_empty, {c: class_to_idx[c] for c in non_empty}
 
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
PngImagePlugin.MAX_TEXT_CHUNK = 16 * (1024**2)  # 16MB


def random_crop_pil(img: Image.Image, tile_size: int, rng: random.Random) -> Image.Image:
    """Sample one random crop of size tile_size x tile_size. Image must be >= tile_size in both dims."""
    w, h = img.size
    x0 = rng.randint(0, w - tile_size)
    y0 = rng.randint(0, h - tile_size)
    return img.crop((x0, y0, x0 + tile_size, y0 + tile_size))


class ZScoreNormalize:
    def __init__(self, eps=1e-6):
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: Tensor [1, H, W]
        mean = x.mean()
        std = x.std()
        return (x - mean) / (std + self.eps)

class RGBDatasetWithAugmentation(Dataset):
    def __init__(self, root='dataset/remote', split='train', augmentation=None, resize=True, size=384, stretch=False):
        
        self.split = split
        self.resize = resize
        self.size = size
        if isinstance(self.size, int):
            self.resize_hw = (self.size, self.size)
        else:
            self.resize_hw = tuple(self.size)
        self.stretch = stretch
        self.rng=random.Random()

        folder_path = os.path.join(root, split)

        self.base_dataset = _FilteredImageFolder(
            root=folder_path,
            transform=None,
        )

        self.to_gray = transforms.Compose([ 
            transforms.Grayscale(num_output_channels=1),
        ])

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            ZScoreNormalize(),
        ])

        self.augmentation = augmentation

    def __getitem__(self, index):
        img, label = self.base_dataset[index]
        
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        tile_size = self.size if isinstance(self.size, int) else min(self.size)
        if self.stretch:
            img = img.resize((tile_size, tile_size), Image.BICUBIC)
        else:
            w, h = img.size
            if w >= tile_size and h >= tile_size:
                img = random_crop_pil(img, tile_size, self.rng)
            else:
                img = img.resize((tile_size, tile_size), Image.BICUBIC)

        img = self.to_gray(img)
        
        if self.split == 'train' and self.augmentation is not None:
            crops = self.augmentation(img)
            img = self.normalize(img)
            return img, crops
        else:
            img = self.normalize(img)
            return img, label
    
    def __len__(self):
        return len(self.base_dataset)
    
    def get_classes(self):
        return self.base_dataset.classes
 
class RGBDatasetLoader:
    def __init__(self, cfg,advance=0, seed=42,  is_dino=True , is_infsampler=True ): 
        assert dist.is_initialized(), "Distributed computing is not initialized! " 

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.data_dir = cfg.dataset.dataset_path
        self.num_workers = cfg.train.num_workers 
        self.resize = getattr(cfg.dataset, "resize", False)
        self.size = getattr(cfg.dataset, "size", (224,224))
        self.stretch = getattr(cfg.dataset, "stretch", False)
        batch_size = getattr(cfg.train, "batch_size_per_gpu", None)
        if batch_size:
            self.batch_size=batch_size
        else: 
            self.global_batch_size = getattr(cfg.train, "global_batch_size", None)

            self.batch_size=self.global_batch_size//self.world_size
         
        
        augmentation=None 
        if is_dino:
            augmentation = DataAugmentationDINO(
                cfg=cfg
            ) 
            
        train_dataset = RGBDatasetWithAugmentation(
            root=self.data_dir,
            split='train',
            augmentation=augmentation,
            size=self.size,
            stretch=self.stretch
        )
        print('train_dataset len: ',len(train_dataset) ) 
        valid_split = 'val'
        if not os.path.exists(os.path.join(self.data_dir, 'valid')) and os.path.exists(os.path.join(self.data_dir, 'test')):
            valid_split = 'test'
            #print(f"'valid' directory not found, using '{valid_split}' directory instead.")
        
        valid_dataset = RGBDatasetWithAugmentation(
            root=self.data_dir,
            split=valid_split,
            size=self.size,
            stretch=self.stretch
        )
        sampler=None
        if is_infsampler: 
            sampler = InfiniteSampler(
                        shuffle=cfg.dataset.shuffle,
                        advance=advance,
                        sample_count=len(train_dataset),
                        seed=seed,
                        rank=self.rank,
                        world_size=self.world_size
                    )
        self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                sampler=sampler
            ) 
         
        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
         
        self.classes = train_dataset.get_classes()
        self.num_classes = len(self.classes)
        
    def get_train_loader(self):
        return self.train_loader
    
    def get_valid_loader(self):
        return self.valid_loader
    
    def get_loaders(self):
        return self.train_loader, self.valid_loader
    
    def get_classes(self):
        return self.classes
    
    def get_num_classes(self):
        return self.num_classes
