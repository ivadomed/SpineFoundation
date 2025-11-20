
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

def plot_6_middle_slices(image: torch.Tensor, gt: torch.Tensor, pred: torch.Tensor):
    """
    Plot 6 middle slices for the image, ground truth (gt), and prediction (pred)
    in a 3-row (I, GT, P), 6-column (slices) grid.
    
    The orientation is assumed to be RPI, slicing along the first dimension (axial/depth).
    It is assumed that the input tensors have a shape like (D, H, W).
    """

    # 1. Bring everything to numpy and float
    # Ensure all inputs are converted to float and numpy arrays for plotting
    image_np = image.float().cpu().numpy()
    gt_np = gt.float().cpu().numpy()
    pred_np = pred.float().cpu().numpy()

    # 2. Determine the central slice and starting index
    depth = image_np.shape[0]
    mid_slice = depth // 2
    # Start 3 slices before the middle (mid_slice - 3)
    start_slice_index = mid_slice - 3
    
    # Check if we have enough slices (6 slices + buffer)
    if start_slice_index < 0 or start_slice_index + 6 > depth:
        print(f"Warning: Not enough slices ({depth} total) to display 6 middle slices. Adjusting start.")
        if depth >= 6:
            start_slice_index = (depth - 6) // 2
        else:
            # Handle case where image is too small
            raise ValueError(f"Image has only {depth} slices, cannot plot 6.")


    # 3. Create the subplot figure
    # 3 rows (Image, GT, Pred) and 6 columns (slices)
    fig, axs = plt.subplots(3, 6, figsize=(18, 9)) # Adjust figsize for better aspect ratio
    fig.suptitle('6 Middle Slices: Image, Ground Truth, and Prediction', fontsize=16)

    # Labels for the rows
    row_labels = ['Input Image', 'Ground Truth (GT)', 'Prediction (Pred)']

    # 4. Loop through the 6 slices
    for col_idx in range(6):
        # Calculate the current slice index
        current_slice_idx = start_slice_index + col_idx
        
        # Extract slices
        slice_image = image_np[current_slice_idx, :, :].T
        slice_gt = gt_np[current_slice_idx, :, :].T
        slice_pred = pred_np[current_slice_idx, :, :].T
        
        # Plotting the 3 rows for the current slice (column)
        
        # Row 0: Image
        ax_img = axs[0, col_idx]
        ax_img.imshow(slice_image, cmap='gray')
        ax_img.axis('off')
        if col_idx == 0:
            ax_img.set_title(row_labels[0], fontsize=12, loc='left')
        ax_img.set_xlabel(f"Slice {current_slice_idx}", fontsize=10) # Label slice index at the bottom row

        # Row 1: Ground Truth
        ax_gt = axs[1, col_idx]
        # Use a distinct colormap like 'jet' or 'viridis' for segmentation masks (GT/Pred) if they are masks.
        # Assuming GT is a segmentation mask (using 'viridis' for a color effect, replace with 'gray' if it's grayscale)
        ax_gt.imshow(slice_gt, cmap='viridis') 
        ax_gt.axis('off')
        if col_idx == 0:
            ax_gt.set_title(row_labels[1], fontsize=12, loc='left')

        # Row 2: Prediction
        ax_pred = axs[2, col_idx]
        ax_pred.imshow(slice_pred, cmap='viridis') # Use the same colormap as GT for consistency
        ax_pred.axis('off')
        if col_idx == 0:
            ax_pred.set_title(row_labels[2], fontsize=12, loc='left')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    plt.show() # Use plt.show() instead of fig.show() in a script environment
    
    return fig
