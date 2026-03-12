from torchvision import transforms
from utils.dino_utils import GaussianBlur
from torchvision.transforms.functional import InterpolationMode
import torch
import random


class AddGaussianNoise:
    """Simule le bruit d'acquisition MRI (Rician ≈ Gaussien à haut SNR)."""
    def __init__(self, std_range=(0.0, 0.1), p=0.5):
        self.std_range = std_range
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            std = random.uniform(*self.std_range)
            return x + torch.randn_like(x) * std
        return x


class ZScoreNormalize:
    def __init__(self, eps=1e-6):
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: Tensor [1, H, W]
        mean = x.mean()
        std = x.std()
        return (x - mean) / (std + self.eps)

class DataAugmentationDINO(object):
    def __init__(self, cfg):

        
        local_crops_number = cfg.crops.local_crops_number
        local_crops_scale = cfg.crops.local_crops_scale
        global_crops_scale = cfg.crops.global_crops_scale
        local_crops_size = cfg.crops.local_crops_size
        global_crops_size = cfg.crops.global_crops_size
        self.local_crops_number = local_crops_number


        # flip_and_color_jitter = transforms.Compose([
        #     transforms.RandomApply(
        #         [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
        #         p=0.8
        #     ),
        # ])

        # intensity = transforms.Compose([
        #     transforms.RandomApply(
        #         [transforms.ColorJitter(brightness=0.2, contrast=0.2)], 
        #         p=0.8
        #     ),
        #     transforms.RandomApply(
        #         [transforms.RandomGamma(gamma=(0.7, 1.5))] if hasattr(transforms, "RandomGamma") else [],
        #         p=0.3
        #     ),
        # ])



        intensity_aug = transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.3, contrast=0.3)],
            p=0.8
        )

        noise_aug = AddGaussianNoise(std_range=(0.0, 0.1), p=0.5)

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomRotation(degrees=15, fill=0),
            intensity_aug,
            transforms.RandomResizedCrop(global_crops_size, scale=global_crops_scale, ratio=(1.0, 1.0), interpolation=InterpolationMode.BICUBIC),
            GaussianBlur(1.0),
            transforms.ToTensor(),
            ZScoreNormalize(),
            noise_aug,
        ])

        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomRotation(degrees=15, fill=0),
            intensity_aug,
            transforms.RandomResizedCrop(global_crops_size, scale=global_crops_scale, ratio=(1.0, 1.0), interpolation=InterpolationMode.BICUBIC),
            GaussianBlur(0.1),
            transforms.ToTensor(),
            ZScoreNormalize(),
            noise_aug,
        ])

        # transformation for the local small crops
        self.local_transfo = transforms.Compose([
            transforms.RandomRotation(degrees=15, fill=0),
            intensity_aug,
            transforms.RandomResizedCrop(local_crops_size, scale=local_crops_scale, ratio=(1.0, 1.0), interpolation=InterpolationMode.BICUBIC),
            GaussianBlur(p=0.5),
            transforms.ToTensor(),
            ZScoreNormalize(),
            noise_aug,
        ])
    
    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        
        # Format output dictionary to match your required structure
        output = {
            "global_crops": [crops[0], crops[1]],
            "local_crops": crops[2:], 
        }
        
        return output