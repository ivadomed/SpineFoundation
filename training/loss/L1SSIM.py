import torch
import torch.nn as nn
from monai.losses import SSIMLoss


class L1_SSIM_Loss(nn.Module):
    """
    pred et target sont déjà normalisés par GPUResampleAug3D._norm (z-score).
    On applique L1 dans l'espace z, et SSIM dans un espace remappé [0,1]
    via des quantiles globaux.
    """
    def __init__(self,
                 alpha: float = 0.85,
                 q_low: float = -1.4036,   # 0.1 %
                 q_high: float = 4.0):     # 99.9 % ou borne sup approchée
        super().__init__()
        assert q_high > q_low
        self.alpha = alpha
        self.q_low = float(q_low)
        self.q_high = float(q_high)

        self.l1 = nn.L1Loss()
        self.ssim3d = SSIMLoss(
            spatial_dims=3,
            data_range=1.0,        # on remappe dans [0,1]
            win_size=5,
            kernel_type="gaussian",
            reduction="mean",
        )

    def _to_01(self, z: torch.Tensor) -> torch.Tensor:
        # clamp dans [q_low, q_high], puis remap linéairement dans [0,1]
        z = torch.clamp(z, self.q_low, self.q_high)
        return (z - self.q_low) / (self.q_high - self.q_low)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Garde-fou : NaN déjà dans les tenseurs d'entrée
        if torch.isnan(pred).any() or torch.isnan(target).any():
            raise RuntimeError("NaN détecté dans pred/target AVANT la loss.")

        pred = pred.float()
        target = target.float()

        # L1 dans l'espace z-score
        l1 = self.l1(pred, target)

        # SSIM dans l'espace borné [0,1]
        pred_n   = self._to_01(pred)
        target_n = self._to_01(target)
        ssim_loss = self.ssim3d(pred_n, target_n)

        loss = self.alpha * l1 + (1.0 - self.alpha) * ssim_loss

        # Si la loss n'est pas finie, on tue l'entraînement avec des logs utiles
        if not torch.isfinite(loss):
            print("=== Non-finite loss détectée dans L1_SSIM_Loss ===")
            print(f"L1 loss       : {l1.detach().cpu().item()}")
            print(f"SSIM loss     : {ssim_loss.detach().cpu().item()}")
            print(f"Loss combinée : {loss.detach().cpu().item()}")
            print(f"pred min/max  : {pred.min().item()} / {pred.max().item()}")
            print(f"tgt  min/max  : {target.min().item()} / {target.max().item()}")
            raise RuntimeError("Non-finite loss dans L1_SSIM_Loss, arrêt du training.")

        return loss
