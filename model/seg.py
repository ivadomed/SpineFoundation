import torch
import torch.nn as nn
import torch.nn.functional as F
from .SpineEncoder import SpineEncoder
class SimpleSegDecoder3D(nn.Module):
    def __init__(self,img_size=(256,256,256),patch_size=(16,16,16),in_dim=768,base_ch=512,dropout=0.0):
        super().__init__()
        self.img_size=tuple(img_size)
        self.patch_size=tuple(patch_size)

        D,H,W=self.img_size
        pd,ph,pw=self.patch_size
        assert D%pd==0 and H%ph==0 and W%pw==0
        self.grid_size=(D//pd,H//ph,W//pw)

        # (B,768,Dg,Hg,Wg) -> (B,512,Dg,Hg,Wg)
        self.proj=nn.Sequential(
            nn.Conv3d(in_dim,base_ch,1,bias=False),
            nn.GroupNorm(16,base_ch),
            nn.GELU()
        )

        # up x2: (Dg,Hg,Wg) -> (2Dg,2Hg,2Wg), channels 512 -> 512
        self.block1=nn.Sequential(
            nn.Conv3d(base_ch,base_ch,3,padding=1,bias=False),
            nn.GroupNorm(16,base_ch),
            nn.GELU()
        )

        # up x2, channels 512 -> 256
        self.block2=nn.Sequential(
            nn.Conv3d(base_ch,base_ch//2,3,padding=1,bias=False),
            nn.GroupNorm(8,base_ch//2),
            nn.GELU()
        )

        # up x2, channels 256 -> 128
        self.block3=nn.Sequential(
            nn.Conv3d(base_ch//2,base_ch//4,3,padding=1,bias=False),
            nn.GroupNorm(8,base_ch//4),
            nn.GELU()
        )

        # final resize to img_size, channels 128 -> 128
        self.block4=nn.Sequential(
            nn.Conv3d(base_ch//4,base_ch//4,3,padding=1,bias=False),
            nn.GroupNorm(8,base_ch//4),
            nn.GELU()
        )

        # binary logits: (B,128,D,H,W) -> (B,1,D,H,W)
        self.head=nn.Conv3d(base_ch//4,1,1)

    def tokens_to_grid(self,z):
        B,N,C=z.shape
        Dg,Hg,Wg=self.grid_size
        assert N == Dg*Hg*Wg
        return z.transpose(1,2).contiguous().view(B,C,Dg,Hg,Wg)

    def forward(self,z,return_probs=False,return_labels=False,threshold=0.5):
        x=self.tokens_to_grid(z)                    # (B,768,Dg,Hg,Wg)
        x=self.proj(x)                              # (B,512,Dg,Hg,Wg)

        x=F.interpolate(x,scale_factor=2,mode="trilinear",align_corners=False)
        x=self.block1(x)

        x=F.interpolate(x,scale_factor=2,mode="trilinear",align_corners=False)
        x=self.block2(x)

        x=F.interpolate(x,scale_factor=2,mode="trilinear",align_corners=False)
        x=self.block3(x)

        x=F.interpolate(x,size=self.img_size,mode="trilinear",align_corners=False)
        x=self.block4(x)

        logits=self.head(x)                         # (B,1,D,H,W)

        if return_labels:
            probs=torch.sigmoid(logits)
            return (probs > threshold).long().squeeze(1)  # (B,D,H,W)

        if return_probs:
            return torch.sigmoid(logits)            # (B,1,D,H,W)

        return logits                               # use with BCEWithLogitsLoss / DiceLoss(sigmoid=True)


class SpineViTSeg(nn.Module):
    def __init__(self,in_channels=1,img_size=(72,276,324),patch_size=(12,12,12),enc_embed_dim=768,enc_num_heads=12,enc_layers=12,enc_mlp_dim=3072,dropout=0.0,mask_ratio=0.0,base_ch=512):
        super().__init__()
        self.encoder=SpineEncoder(in_channels=in_channels,img_size=img_size,patch_size=patch_size,enc_embed_dim=enc_embed_dim,enc_num_heads=enc_num_heads,enc_layers=enc_layers,enc_mlp_dim=enc_mlp_dim,dropout=dropout,mask_ratio=mask_ratio)
        self.decoder=SimpleSegDecoder3D(img_size=img_size,patch_size=patch_size,in_dim=enc_embed_dim,base_ch=base_ch,dropout=dropout)

    def forward(self,x,return_probs=False,return_labels=False,threshold=0.5):
        z,_=self.encoder(x)
        return self.decoder(z,return_probs=return_probs,return_labels=return_labels,threshold=threshold)
