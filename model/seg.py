import torch
import torch.nn as nn
import torch.nn.functional as F

from model.SpineEncoder import SpineEncoder


class SimpleSegDecoder3D(nn.Module):
    def __init__(self,img_size=(256,256,256),patch_size=(16,16,16),in_dim=768,num_classes=2,base_ch=256,dropout=0.0):
        super().__init__()
        self.img_size=tuple(img_size)
        self.patch_size=tuple(patch_size)

        D,H,W=self.img_size
        pd,ph,pw=self.patch_size
        assert D%pd==0 and H%ph==0 and W%pw==0

        self.grid_size=(D//pd,H//ph,W//pw)

        self.proj=nn.Sequential(nn.Conv3d(in_dim,base_ch,1,bias=False),nn.GroupNorm(8,base_ch),nn.GELU())
        self.block1=nn.Sequential(nn.Conv3d(base_ch,base_ch,3,padding=1,bias=False),nn.GroupNorm(8,base_ch),nn.GELU())
        self.block2=nn.Sequential(nn.Conv3d(base_ch,base_ch//2,3,padding=1,bias=False),nn.GroupNorm(8,base_ch//2),nn.GELU())
        self.block3=nn.Sequential(nn.Conv3d(base_ch//2,base_ch//4,3,padding=1,bias=False),nn.GroupNorm(8,base_ch//4),nn.GELU())

        self.head=nn.Conv3d(base_ch//4,num_classes,1)

    def tokens_to_grid(self,z):
        B,N,C=z.shape
        Dg,Hg,Wg=self.grid_size
        z=z.transpose(1,2).contiguous().view(B,C,Dg,Hg,Wg)
        return z

    def forward(self,z):
        x=self.tokens_to_grid(z)
        x=self.proj(x)
        x=F.interpolate(x,scale_factor=2,mode="trilinear",align_corners=False)
        x=self.block1(x)
        x=F.interpolate(x,scale_factor=2,mode="trilinear",align_corners=False)
        x=self.block2(x)
        x=F.interpolate(x,size=self.img_size,mode="trilinear",align_corners=False)
        x=self.block3(x)
        return self.head(x)


class SpineViTSeg(nn.Module):
    def __init__(self,in_channels=1,img_size=(72,72,324),patch_size=(12,12,12),enc_embed_dim=768,enc_num_heads=12,enc_layers=12,enc_mlp_dim=3072,dropout=0.0,mask_ratio=0.0,num_classes=2):
        super().__init__()

        self.encoder=SpineEncoder(in_channels=in_channels,img_size=img_size,patch_size=patch_size,enc_embed_dim=enc_embed_dim,enc_num_heads=enc_num_heads,enc_layers=enc_layers,enc_mlp_dim=enc_mlp_dim,dropout=dropout,mask_ratio=mask_ratio)

        self.decoder=SimpleSegDecoder3D(img_size=img_size,patch_size=patch_size,in_dim=enc_embed_dim,num_classes=num_classes)

    def forward(self,x):
        z,_=self.encoder(x)   # ids_restore
        logits=self.decoder(z)
        return logits