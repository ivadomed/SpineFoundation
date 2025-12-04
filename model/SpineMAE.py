"""
This script contains the SpineMAE model class.

Author: Thomas Dagonneau & Julien Laborde-Peyré
"""

import torch
import torch.nn as nn

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from model.SpineEncoder import SpineEncoder
from model.SpineDecoder import SpineDecoder
    


class SpineMAE(nn.Module):
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
            return self.decoder.forward(z, ids_restore)


            
if __name__ == "__main__":

    img_size = (32, 32, 32)
    patch_size = (8, 8, 8)

    enc = SpineEncoder(in_channels=1, img_size=img_size, patch_size=patch_size,
                        enc_embed_dim=100, enc_num_heads=4, enc_layers=2, enc_mlp_dim=128, dropout=0.0)
    dec = SpineDecoder(img_size=img_size, patch_size=patch_size, enc_embed_dim=100,dec_mlp_dim=128,
                        dec_embed_dim=32, dec_layers=1, dec_num_heads=4, in_channels=1)


    x = torch.randn(1, 1, *img_size)
    z,ids_restore = enc.forward(x)
    recon = dec.forward(z, ids_restore)
    print('\nTaille:')
    print('z', z.shape)
    print('recon', recon.shape)



    


    from torchinfo import summary as torch_summary

    print('\nEncoder summary:')
    torch_summary(enc, input_size=(1, 1, *img_size))

    print('\nEncoder+Decoder summary:')
    wrapper = SpineMAE(img_size = (32, 32, 32),patch_size = (8, 8, 8))
    torch_summary(wrapper, input_size=(1, 1, *img_size))

        