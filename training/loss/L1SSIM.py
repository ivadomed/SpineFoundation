import torch 
import torch.nn as nn 
import torch 
import torch.nn as nn 
from monai.losses import SSIMLoss


class L1_SSIM_Loss(nn.Module):
    def __init__(self, alpha=0.85, data_range=1.0):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        self.ssim3d = SSIMLoss(
            spatial_dims=3,
            data_range=data_range,
            win_size=5,              # au lieu de 11
            kernel_type="gaussian",  # ou "uniform" encore un peu plus simple
            reduction="mean",
        )

    def forward(self, pred, target):
        l1 = self.l1(pred, target)
        ssim_loss = self.ssim3d(pred, target)
        return self.alpha * l1 + (1.0 - self.alpha) * ssim_loss
