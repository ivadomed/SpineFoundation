import torch
import torch.nn as nn
import pytorch_msssim

class L1_SSIM_Loss(nn.Module):
    def __init__(self, alpha=0.85):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        self.ssim = pytorch_msssim.SSIM(data_range=1.0, size_average=True)

    def forward(self, pred, target):
        l1 = self.l1(pred, target)
        ssim = 1 - self.ssim(pred, target)
        return self.alpha * l1 + (1-self.alpha) * ssim
