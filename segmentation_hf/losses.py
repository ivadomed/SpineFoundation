import torch


def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.reshape(probs.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def compute_dice_score(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    preds = (torch.sigmoid(logits) > 0.5).float()
    preds = preds.reshape(preds.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    denom = preds.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return dice.mean().item()
