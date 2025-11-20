"""
This script contains the SpineEncoder model class.

Author: Thomas Dagonneau & Julien Laborde-Peyré
"""
import torch
import torch.nn as nn

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock

    

def random_masking(x, mask_ratio):
    B, N, C = x.shape
    num_keep = int(N * (1 - mask_ratio))

    bruit=torch.rand(B, N, device=x.device)
    ids_shuffle=torch.argsort(bruit, dim=1) 
    ids_restore=torch.argsort(ids_shuffle, dim=1)

    ids_keep=ids_shuffle[:, :num_keep] 

    # sélectionner les tokens visibles
    x_visible = torch.gather(
        x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, C)
    )

    return x_visible,ids_restore

class SpineEncoder(nn.Module):
    def __init__(self,in_channels=1,img_size=(256, 256, 256),patch_size=(16, 16, 16),
            embed_dim=100,num_heads=12,num_layers=12,mlp_dim=3072,dropout_rate=0,mask_ratio=0):
        super().__init__()
        self.mask_ratio = mask_ratio    
        self.patch_embedding = PatchEmbeddingBlock(in_channels=in_channels,img_size=img_size,patch_size=patch_size,
        hidden_size=embed_dim,num_heads=num_heads,proj_type="conv",dropout_rate=dropout_rate,spatial_dims=3)

        self.transformer_layers = nn.ModuleList([TransformerBlock(
                hidden_size=embed_dim,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate
            ) for k in range(num_layers)]) 

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embedding(x)     # (B, N, embeddim)
        x_visible, ids_restore =random_masking(x, self.mask_ratio) 

        z = x_visible
        for blk in self.transformer_layers:
            z = blk(z)
        z = self.norm(z)                        
        return z, ids_restore