"""
This script contains the SpineMAE model class.

Author: Thomas Dagonneau & Julien Laborde-Peyré
"""

import torch
import torch.nn as nn

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
    
class SpineDecoder(nn.Module):
    def __init__(self,img_size=(256, 256,256),patch_size=(16, 16, 16),enc_embed_dim=256,
        dec_embed_dim=128,dec_layers= 4,dec_num_heads= 4,in_channels=1,dec_mlp_dim=3072):
        super().__init__()

        num_patches = img_size[0] // patch_size[0] * img_size[1] // patch_size[1] * img_size[2] // patch_size[2]

        # Pour la reconstruction
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = num_patches



        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim)    
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_embed_dim)) #Represente l'ensemble des tokens masqués
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, dec_embed_dim))

        self.blocks = nn.ModuleList([TransformerBlock(
                    hidden_size=dec_embed_dim,mlp_dim=dec_mlp_dim,
                    num_heads=dec_num_heads,
                ) for k in range(dec_layers)])



        self.norm = nn.LayerNorm(dec_embed_dim)
        patch_voxels = patch_size[0] * patch_size[1] * patch_size[2] * in_channels
        self.pred = nn.Linear(dec_embed_dim, patch_voxels) #Normalisation puis prédiction de l'image

        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)


    def embed(self, x_visible, ids_restore):
        B, N_vis,embed_dim = x_visible.shape 
        N = ids_restore.shape[1] #Nombre de patchs total
        x = self.decoder_embed(x_visible)
        N_mask = N - N_vis

    
        mask_tokens = self.mask_token.expand(B, N_mask, -1)
        x = torch.cat([x, mask_tokens], dim=1)

        x = torch.gather(x,1,ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[-1])) #remet dans l'ordre correct

        x = x + self.decoder_pos_embed
        return x

    def unpatchify(self, x_patches):

        B, N, pv = x_patches.shape
        pD, pH, pW = self.patch_size
        C = self.in_channels

        # Nombre de patchs suivant chaque axe
        Dp = self.img_size[0] // pD
        Hp = self.img_size[1] // pH
        Wp = self.img_size[2] // pW


        x = x_patches.view(B, Dp, Hp, Wp, C, pD, pH, pW)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)
        x = x.reshape(B, C, Dp * pD, Hp * pH, Wp * pW)
        return x

    def forward(self, x_visible, ids_restore):
        
        x = self.embed(x_visible, ids_restore)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        pred = self.pred(x)  # (B, N, patch_voxels)
        recon = self.unpatchify(pred)
        return recon