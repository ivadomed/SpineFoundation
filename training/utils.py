
from typing import Tuple
import os
import torch
import json
import matplotlib.pyplot as plt

def list_child_folders(path: str):
    print(path)
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

def load_json_param(param):

    if param.endswith(".json") and os.path.isfile(param):
        with open(param, "r") as f:
            return json.load(f)

    return json.loads(param)

def plot_slices(image):
    """
    Plot the image, ground truth and prediction of the mid-sagittal axial slice
    The orientaion is assumed to RPI
    """

    # bring everything to numpy 
    ## added the .float() because of issue : TypeError: Got unsupported ScalarType BFloat16
    image = image.float().numpy()
    

    mid_sagittal = image.shape[0]//2
    # plot X slices before and after the mid-sagittal slice in a grid
    fig, axs = plt.subplots(1, 6, figsize=(18, 54))
    fig.suptitle('Original Image')
    for i in range(6):
        axs[i].imshow(image[mid_sagittal-3+i,:,:].T, cmap='gray'); axs[i].axis('off') 

    plt.tight_layout()
    fig.show()
    
    return fig
