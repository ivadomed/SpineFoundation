# mae3d_monai.py
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from monai.data import Dataset
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
    RandFlipd, RandAffined, SpatialPadd, Resized, Compose, ResizeWithPadOrCropd
)

# -------------------------------------------------------
# 3D Patch Embedding
# -------------------------------------------------------

class PatchEmbed3D(nn.Module):
    """Split a 3D volume into non-overlapping patches and embed them."""
    def __init__(self, in_chans, patch_size, embed_dim, img_size):
        super().__init__()
        self.patch_size = patch_size

        assert all(img_size[i] % patch_size[i] == 0 for i in range(3)), \
            "img_size must be divisible by patch_size in all dimensions"

        self.num_patches = (
            (img_size[0] // patch_size[0]) *
            (img_size[1] // patch_size[1]) *
            (img_size[2] // patch_size[2])
        )

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, D, H, W)
        x = self.proj(x)  # (B, E, D/p, H/p, W/p)
        B, E, Dp, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, E)
        return x


# -------------------------------------------------------
# Transformer Encoder (shared for encoder and decoder)
# -------------------------------------------------------

class Transformer(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleDict({
                    "norm1": nn.LayerNorm(embed_dim),
                    "attn": nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                    "norm2": nn.LayerNorm(embed_dim),
                    "mlp": nn.Sequential(
                        nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                        nn.GELU(),
                        nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
                    )
                })
            )

    def forward(self, x):
        for layer in self.layers:
            attn_out, _ = layer["attn"](layer["norm1"](x), x, x)
            x = x + attn_out
            x = x + layer["mlp"](layer["norm2"](x))
        return x


# -------------------------------------------------------
# Masked Autoencoder for 3D MRI
# -------------------------------------------------------

class MAE3D(nn.Module):
    """
    3D Masked Autoencoder (MAE) for MRI.
    Implements the *true MAE* behavior:
    - Encoder sees only visible tokens.
    - Decoder reconstructs using visible + mask tokens.
    """
    def __init__(
        self,
        img_size=(128, 128, 128),
        patch_size=(16, 16, 16),
        in_chans=1,
        embed_dim=256,
        encoder_depth=8,
        encoder_heads=8,
        decoder_embed_dim=128,
        decoder_depth=4,
        decoder_heads=4,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans

        # Patch embedding
        self.patch_embed = PatchEmbed3D(in_chans, patch_size, embed_dim, img_size)
        self.num_patches = self.patch_embed.num_patches

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Encoder
        self.encoder = Transformer(embed_dim, encoder_depth, encoder_heads)

        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim))

        self.decoder = Transformer(decoder_embed_dim, decoder_depth, decoder_heads)

        patch_voxels = patch_size[0] * patch_size[1] * patch_size[2] * in_chans
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_voxels)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    # ---------------------------------------------------
    # Masking: random permutation + take first k%
    # ---------------------------------------------------

    def random_mask(self, B, mask_ratio, device):
        N = self.num_patches
        num_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :num_keep]

        mask = torch.ones(B, N, device=device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, 1, ids_shuffle)

        return ids_keep, mask, ids_shuffle

    # ---------------------------------------------------
    # Forward: true MAE behavior
    # ---------------------------------------------------

    def forward(self, x, mask_ratio=0.75):
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, E)
        x = x + self.pos_embed

        ids_keep, mask, ids_shuffle = self.random_mask(B, mask_ratio, x.device)

        # Select visible tokens
        batch_indices = torch.arange(B).unsqueeze(-1)
        x_visible = x[batch_indices, ids_keep]  # (B, N_visible, E)

        # Encoder
        encoded = self.encoder(x_visible)

        # Prepare decoder input
        dec_tokens = self.decoder_embed(encoded)

        mask_tokens = self.mask_token.repeat(B, self.num_patches - dec_tokens.shape[1], 1)

        # Concatenate visible + masked tokens in correct order
        dec_all = torch.cat([dec_tokens, mask_tokens], dim=1)

        # Unshuffle
        dec_all = torch.gather(dec_all, 1, ids_shuffle.unsqueeze(-1).expand(-1, -1, dec_all.shape[-1]))

        dec_all = dec_all + self.decoder_pos_embed

        decoded = self.decoder(dec_all)

        pred = self.decoder_pred(decoded)
        pred = pred.view(
            B,
            self.num_patches,
            self.in_chans,
            self.patch_size[0],
            self.patch_size[1],
            self.patch_size[2]
        )
        return pred, mask


# -------------------------------------------------------
# Patchify / Unpatchify
# -------------------------------------------------------

def patchify3d(vol, patch_size):
    B, C, D, H, W = vol.shape
    pD, pH, pW = patch_size

    vol = vol.view(
        B, C,
        D // pD, pD,
        H // pH, pH,
        W // pW, pW
    )
    vol = vol.permute(0, 2, 4, 6, 1, 3, 5, 7)
    patches = vol.reshape(B, -1, C, pD, pH, pW)
    return patches


# -------------------------------------------------------
# Training Loop
# -------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, patch_size, epoch, mask_ratio=0.75):
    model.train()
    loss_fn = nn.MSELoss()
    total_loss = 0

    for batch in loader:
        img = batch["image"].float().to(device)  # (B,1,D,H,W)

        # model returns:
        # pred:  (B, N, P, P, P)
        # mask:  (B, N)
        pred, mask = model(img, mask_ratio)

        # target: (B, N, P, P, P)
        target = patchify3d(img, patch_size)

        mask = mask.to(device=pred.device, dtype=torch.bool)

        # mask: [B, N]
        # make it [B, N, 1, 1, 1, 1]
        mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # expand to match voxel grid inside each patch
        P = pred.shape[-1]  # patch_size (e.g. 16)
        mask = mask.expand(-1, -1, 1, P, P, P)

        loss = loss_fn(pred[mask], target[mask])


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}: loss={total_loss/len(loader):.4f}")



# -------------------------------------------------------
# Example usage with MONAI data pipeline
# -------------------------------------------------------

def main():
    # Example: 128x128x128 MRI, single channel
    img_size = (128, 128, 128)
    patch_size = (16, 16, 16)

    # Data
    data_dir = "../imagesTs"
    files = [{"image": os.path.join(data_dir, f)} for f in os.listdir(data_dir) if f.endswith(".nii.gz")]
    print(files)
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]), 
        ResizeWithPadOrCropd(keys=["image"], spatial_size=img_size),
        ScaleIntensityd(keys=["image"]), 
        #SpatialPadd(keys=["image"], spatial_size=img_size), #J'ai  l'impression que c'est redondant avec ResizeWithPadOrCropd
        #Resized(keys=["image"], spatial_size=img_size), #En contradiction avec ResizeWithPadOrCropd, ici de l'interpolation est faite pour déformer l'image à spatial_size   
        RandFlipd(keys=["image"], spatial_axis=0, prob=0.2), 
        RandAffined(keys=["image"], rotate_range=0.1, prob=0.2),  
    ])

    ds = Dataset(files, transforms)
    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=4) 

    # Model
    model = MAE3D(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=1,
        embed_dim=256,
        encoder_depth=8,
        encoder_heads=8,
        decoder_embed_dim=128,
        decoder_depth=4,
        decoder_heads=4
    ).cuda()

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Train
    for epoch in range(20):
        train_one_epoch(model, loader, optimizer, "cuda", patch_size, epoch)


if __name__ == "__main__":
    main()
