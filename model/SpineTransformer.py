"""
This script contains the SpineTransformer model class.

Author: Thomas Dagonneau & Julien Laborde-Peyré
Architecture inspiration : https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
"""

import torch
import torch.nn as nn

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock



class SpineEncoder(nn.Module):
    """
    SpineTransformer model for medical image analysis.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        img_size (tuple): Size of the input image (H, W).
        patch_size (tuple): Size of the patches (H, W).
        embed_dim (int): Dimension of the embedding.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of transformer layers.
        mlp_dim (int): Dimension of the MLP in the transformer block.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self,
                 in_channels=1,
                 out_channels=2,
                 img_size=(256, 256),
                 patch_size=(16, 16),
                 embed_dim=768,
                 num_heads=12,
                 num_layers=12,
                 mlp_dim=3072,
                 dropout_rate=0.1):
        super().__init__()

        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=embed_dim,
            num_heads=num_heads,
            pos_embed="conv",           # ou "perceptron"
            dropout_rate=dropout_rate,
            spatial_dims=3,             
        )


        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=embed_dim,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate
            ) for _ in range(num_layers)
        ]) 

    def forward(self, x):
        x = self.patch_embedding(x)
        for layer in self.transformer_layers:
            x = layer(x)
        return x


    

class SpineDecoder(nn.Module):
    def __init__(
        self,
        num_patches: int,
        embed_dim: int = 256,
        decoder_embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        patch_size: tuple = (16, 16, 16),
        in_channels: int = 1,
    ):
        super().__init__()

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)    
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_e  mbed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim))

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=decoder_embed_dim,
                    mlp_dim=4 * decoder_embed_dim,
                    num_heads=num_heads,
                )
                for _ in range(depth)
            ]
        )



        self.norm = nn.LayerNorm(decoder_embed_dim)
        patch_voxels = patch_size[0] * patch_size[1] * patch_size[2] * in_channels
        self.pred = nn.Linear(decoder_embed_dim, patch_voxels) #Normalisation puis prédiction de l'image

        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)

    def forward(self, x_visible, ids_restore):
        B, N_vis, _ = x_visible.shape 
        N = ids_restore.shape[1] #Normalement ~80 % des tokens sont masqués

        x = self.decoder_embed(x_visible)
        N_mask = N - N_vis

        if N_mask > 0:
            mask_tokens = self.mask_token.expand(B, N_mask, -1)
            x = torch.cat([x, mask_tokens], dim=1)

        x = torch.gather(
            x,
            1,
            ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[-1]),
        )

        x = x + self.decoder_pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return self.pred(x)