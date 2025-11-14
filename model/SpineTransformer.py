"""
This script contains the SpineTransformer model class.

Author: Thomas Dagonneau & Julien Laborde-Peyré
"""

import torch
import torch.nn as nn

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock

class SpineTransformer(nn.Module):
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
        super(SpineTransformer, self).__init__()

        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_size=img_size,
            norm_layer=nn.LayerNorm
        )

        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout_rate=dropout_rate
            ) for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_channels)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        for layer in self.transformer_layers:
            x = layer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.classifier(x)
        return x


class Transformer(nn.Module):
    

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