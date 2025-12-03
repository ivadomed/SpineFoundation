
from typing import Tuple
import os
import torch
import json
import matplotlib.pyplot as plt
import numpy as np

def list_child_folders(path: str):
    print("Path utilisé:", path)
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

def plot_6_middle_slices(image: torch.Tensor, gt: torch.Tensor, pred: torch.Tensor):
    def mask_patches_2d(slice_2d, patch_size=16, mask_ratio=0.75):
        """
        Découpe slice_2d en patches patch_size x patch_size et met à zéro
        mask_ratio * 100 % des patches (tirage aléatoire).
        """
        slice_2d = slice_2d.copy()
        H, W = slice_2d.shape
        ph, pw = patch_size, patch_size

        n_ph = H // ph  # nombre de patches complets en hauteur
        n_pw = W // pw  # nombre de patches complets en largeur
        total_patches = n_ph * n_pw

        if total_patches == 0:
            return slice_2d  # image trop petite

        # True -> patch masqué (mis à zéro)
        mask = np.random.rand(total_patches) < mask_ratio

        idx = 0
        for i in range(n_ph):
            for j in range(n_pw):
                if mask[idx]:
                    h0, h1 = i * ph, (i + 1) * ph
                    w0, w1 = j * pw, (j + 1) * pw
                    slice_2d[h0:h1, w0:w1] = 0.0
                idx += 1

        return slice_2d

    # 1. Bring everything to numpy and float
    image_np = image.float().cpu().detach().numpy()
    gt_np = gt.float().cpu().detach().numpy()
    pred_np = pred.float().cpu().detach().numpy()

    # 2. Determine the central slice and starting index
    depth = image_np.shape[0]
    mid_slice = depth // 2
    start_slice_index = mid_slice - 3

    if start_slice_index < 0 or start_slice_index + 6 > depth:
        print(f"Warning: Not enough slices ({depth} total) to display 6 middle slices. Adjusting start.")
        if depth >= 6:
            start_slice_index = (depth - 6) // 2
        else:
            raise ValueError(f"Image has only {depth} slices, cannot plot 6.")

    # 3. Create the subplot figure
    fig, axs = plt.subplots(3, 6, figsize=(18, 9))
    fig.suptitle('6 Middle Slices: Image (patched+masked), Ground Truth, and Prediction', fontsize=16)

    row_labels = ['Input Image (75% patches masked)', 'Ground Truth (GT)', 'Prediction (Pred)']

    # 4. Loop through the 6 slices
    for col_idx in range(6):
        current_slice_idx = start_slice_index + col_idx

        # extractions (depth, H, W) puis transpose pour correspondre à ton affichage actuel
        slice_image = image_np[current_slice_idx, :, :].T
        slice_gt = gt_np[current_slice_idx, :, :].T
        slice_pred = pred_np[current_slice_idx, :, :].T

        # --- patching + masking 75% des patches 16x16 sur l'input uniquement ---
        slice_image_masked = mask_patches_2d(slice_image, patch_size=16, mask_ratio=0.75)

        # Row 0: Image masquée
        ax_img = axs[0, col_idx]
        ax_img.imshow(slice_image_masked, cmap='gray')
        ax_img.axis('off')
        if col_idx == 0:
            ax_img.set_title(row_labels[0], fontsize=12, loc='left')
        ax_img.set_xlabel(f"Slice {current_slice_idx}", fontsize=10)

        # Row 1: Ground Truth
        ax_gt = axs[1, col_idx]
        ax_gt.imshow(slice_gt, cmap='gray')
        ax_gt.axis('off')
        if col_idx == 0:
            ax_gt.set_title(row_labels[1], fontsize=12, loc='left')

        # Row 2: Prediction
        ax_pred = axs[2, col_idx]
        ax_pred.imshow(slice_pred, cmap='gray')
        ax_pred.axis('off')
        if col_idx == 0:
            ax_pred.set_title(row_labels[2], fontsize=12, loc='left')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    
    return fig

def plot_6_uniform_slices(image: torch.Tensor, gt: torch.Tensor, pred: torch.Tensor):

    # 1. Convertir en numpy
    image_np = image.float().cpu().detach().numpy()   # [D, H, W]
    gt_np = gt.float().cpu().detach().numpy()
    pred_np = pred.float().cpu().detach().numpy()

    depth, H, W = image_np.shape

    if depth < 8:
        raise ValueError(f"Image trop petite ({depth} slices). Min = 8 pour une division en 8.")

    # 2. Mask 75% de l’image input, mais au niveau des patches 8x8
    patch_h, patch_w = 8, 8
    nph = (H + patch_h - 1) // patch_h  # ceil(H / 8)
    npw = (W + patch_w - 1) // patch_w  # ceil(W / 8)

    # mask de patches : True = garder, False = masquer
    # proba de garder ~25% -> ~75% masqué
    patch_keep = np.random.rand(nph, npw) > 0.75  # [nph, npw]

    # Étendre chaque patch en un bloc 16x16
    patch_keep_big = np.kron(patch_keep, np.ones((patch_h, patch_w), dtype=bool))  # [nph*16, npw*16]

    # Rogner à la taille exacte H, W
    patch_keep_big = patch_keep_big[:H, :W]  # [H, W]

    # Étendre sur la profondeur (même mask pour toutes les slices)
    mask = np.broadcast_to(patch_keep_big, (depth, H, W))  # [D, H, W]

    # Appliquer le mask : les patches masqués deviennent 0
    masked_image_np = image_np * mask

    # 3. Slices uniformément réparties : on divise la profondeur en 8 segments et on prend 6 indices internes
    step = depth // 15
    slice_indices = [step * i for i in range(5, 11)]  # indices 5..10

    # 4. Figure (3 rows × 6 columns)
    fig, axs = plt.subplots(3, 6, figsize=(18, 9))
    fig.suptitle('6 Uniform Slices: Masked Input (patch 16x16), GT, Prediction', fontsize=16)

    row_labels = ['Masked Input (75%)', 'Ground Truth (GT)', 'Prediction (Pred)']

    # 5. Plot
    for col_idx, slice_idx in enumerate(slice_indices):

        slice_img  = masked_image_np[slice_idx, :, :].T
        slice_gt   = gt_np[slice_idx, :, :].T
        slice_pred = pred_np[slice_idx, :, :].T

        # Row 0 : masked input
        ax = axs[0, col_idx]
        ax.imshow(slice_img, cmap='gray')
        ax.axis('off')
        if col_idx == 0:
            ax.set_title(row_labels[0], fontsize=12, loc='left')
        ax.set_xlabel(f"Slice {slice_idx}", fontsize=10)

        # Row 1 : GT
        ax = axs[1, col_idx]
        ax.imshow(slice_gt, cmap='gray')
        ax.axis('off')
        if col_idx == 0:
            ax.set_title(row_labels[1], fontsize=12, loc='left')

        # Row 2 : Pred
        ax = axs[2, col_idx]
        ax.imshow(slice_pred, cmap='gray')
        ax.axis('off')
        if col_idx == 0:
            ax.set_title(row_labels[2], fontsize=12, loc='left')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    return fig


