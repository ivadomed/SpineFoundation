"""
This script contains the SpineMAE model class.

Author: Thomas Dagonneau & Julien Laborde-Peyré
"""

import torch
import torch.nn as nn

from monai.networks.blocks.transformerblock import TransformerBlock
from model.SpineEncoder import SpineEncoder

class SpineDecoder(nn.Module):
    def __init__(self, img_size=(256,256,256), patch_size=(16,16,16), enc_embed_dim=256, dec_embed_dim=128, dec_layers=4, dec_num_heads=4, in_channels=1, dec_mlp_dim=3072):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels

        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        self.num_patches = num_patches

        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, dec_embed_dim))

        self.blocks = nn.ModuleList([TransformerBlock(hidden_size=dec_embed_dim, mlp_dim=dec_mlp_dim, num_heads=dec_num_heads) for _ in range(dec_layers)])
        self.norm = nn.LayerNorm(dec_embed_dim)

        patch_voxels = patch_size[0] * patch_size[1] * patch_size[2] * in_channels
        self.pred = nn.Linear(dec_embed_dim, patch_voxels)

        nn.init.normal_(self.decoder_pos_embed, std=0.02)

    def embed(self, x_tokens):
        # x_tokens: (B, N, enc_embed_dim) avec N = num_patches
        x = self.decoder_embed(x_tokens)                       # (B, N, dec_embed_dim)
        x = x + self.decoder_pos_embed[:, : x.shape[1], :]     # supporte N variable si besoin
        return x

    def unpatchify(self, x_patches):
        # x_patches: (B, N, patch_voxels)
        B, N, pv = x_patches.shape
        pD, pH, pW = self.patch_size
        C = self.in_channels

        Dp = self.img_size[0] // pD
        Hp = self.img_size[1] // pH
        Wp = self.img_size[2] // pW
        assert N == Dp * Hp * Wp, f"N={N} != Dp*Hp*Wp={Dp*Hp*Wp}"

        x = x_patches.view(B, Dp, Hp, Wp, C, pD, pH, pW)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        x = x.view(B, C, Dp * pD, Hp * pH, Wp * pW)
        return x

    def forward(self, x_tokens):
        # x_tokens: (B, num_patches, enc_embed_dim) (pas de mask, pas de ids_restore)
        x = self.embed(x_tokens)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        pred = self.pred(x)          # (B, N, patch_voxels)
        recon = self.unpatchify(pred)
        return recon



class SpineSeg(nn.Module):

    def __init__(self,in_channels=1,img_size=(256, 256, 256),patch_size=(16, 16, 16),
        enc_embed_dim=120,enc_num_heads=12,enc_layers=12,enc_mlp_dim=3072,dropout=0,mask_ratio=0,
        dec_embed_dim=128,dec_layers= 4,dec_num_heads= 4,dec_mlp_dim=3072):
        super().__init__()
        self.encoder = SpineEncoder(in_channels=1, img_size=img_size, patch_size=patch_size,
                    enc_embed_dim=enc_embed_dim, enc_num_heads=enc_num_heads, enc_layers=enc_layers, enc_mlp_dim=enc_mlp_dim, dropout=dropout,mask_ratio=mask_ratio)
        self.decoder = SpineDecoder(img_size=img_size, patch_size=patch_size, enc_embed_dim=enc_embed_dim,dec_mlp_dim=dec_mlp_dim,
                    dec_embed_dim=dec_embed_dim, dec_layers=dec_layers, dec_num_heads=dec_num_heads, in_channels=in_channels)

    def forward(self, x):
        z, ids_restore = self.encoder.forward(x)
        return self.decoder.forward(z)