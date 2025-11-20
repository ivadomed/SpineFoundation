
from typing import Tuple
import os
import torch

def list_child_folders(path: str):

    if not os.path.isdir(path):
        raise NotADirectoryError(f"{path} n'est pas un dossier valide")

    return [
        os.path.join(path, name) for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))
    ]


def patchify(x: torch.Tensor, patch_size: Tuple[int, int, int]) -> torch.Tensor:
    B, C, D, H, W = x.shape
    pD, pH, pW = patch_size
    assert D % pD == 0 and H % pH == 0 and W % pW == 0
    Dp = D // pD
    Hp = H // pH
    Wp = W // pW

    x = x.view(B, C, Dp, pD, Hp, pH, Wp, pW)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)  # B, Dp, Hp, Wp, C, pD, pH, pW
    x = x.contiguous().view(B, Dp * Hp * Wp, C * pD * pH * pW)
    return x


def save_checkpoint(state, filename: str):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(path: str, device='cpu'):
    return torch.load(path, map_location=device)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
