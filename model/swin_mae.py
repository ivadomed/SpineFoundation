# swin_mae_3d_no_dropout.py
# Swin-MAE 3D — MODEL ONLY (no preprocessing, NO DROPOUT, NO DROPPATH)
#
# What is included:
# - 3D Patch Embedding (Conv3d)
# - Swin blocks 3D (window attention + shifted windows)
# - Patch Merging 3D (encoder hierarchy)
# - Patch Expanding 3D (decoder mirror)
# - Learned relative position bias (Δd, Δh, Δw)
# - Block masking on TOKEN grid
# - MAE reconstruction head (patchified voxel target)
#
# What is NOT included:
# - Any preprocessing
# - Any data loading
# - Any dropout / stochastic depth
#
# Assumptions:
# - Input x: (B, C, D, H, W)
# - (D,H,W) divisible by patch_size
# - Token grid divisible by window_size
# - Token grid divisible by 2**(num_stages-1)

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# utils
# ----------------------------
def _ntuple3(x):
    if isinstance(x, (tuple, list)):
        return (int(x[0]), int(x[1]), int(x[2]))
    return (int(x), int(x), int(x))


class Mlp(nn.Module): #multilayer perceptron
    def __init__(self, dim: int, mlp_ratio: float):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# ----------------------------
# window ops (3D)
# ----------------------------
def window_partition_3d(x, window_size):
    wd, wh, ww = window_size
    B, D, H, W, C = x.shape
    x = x.contiguous().reshape(B, D // wd, wd, H // wh, wh, W // ww, ww, C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return x.reshape(-1, wd, wh, ww, C)



def window_reverse_3d(windows, window_size, D, H, W):
    wd, wh, ww = window_size
    B = int(windows.shape[0] // ((D // wd) * (H // wh) * (W // ww)))
    x = windows.contiguous().reshape(B, D // wd, H // wh, W // ww, wd, wh, ww, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.reshape(B, D, H, W, -1)


# ----------------------------
# relative position bias 3D
# ----------------------------
class RelativePositionBias3D(nn.Module):
    def __init__(self, window_size, num_heads):
        super().__init__()
        wd, wh, ww = window_size
        size = (2 * wd - 1) * (2 * wh - 1) * (2 * ww - 1)
        self.table = nn.Parameter(torch.zeros(size, num_heads))
        nn.init.trunc_normal_(self.table, std=0.02)

        coords = torch.stack(torch.meshgrid(
            torch.arange(wd),
            torch.arange(wh),
            torch.arange(ww),
            indexing="ij"
        ))
        coords = coords.flatten(1)
        rel = coords[:, :, None] - coords[:, None, :]
        rel = rel.permute(1, 2, 0)
        rel[..., 0] += wd - 1
        rel[..., 1] += wh - 1
        rel[..., 2] += ww - 1
        rel[..., 0] *= (2 * wh - 1) * (2 * ww - 1)
        rel[..., 1] *= (2 * ww - 1)
        self.register_buffer("index", rel.sum(-1), persistent=False)

    def forward(self):
        N = self.index.shape[0]
        return self.table[self.index.view(-1)].view(N, N, -1).permute(2, 0, 1)


# ----------------------------
# window attention 3D
# ----------------------------
class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.rpb = RelativePositionBias3D(window_size, num_heads)

    def forward(self, x, attn_mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn + self.rpb().unsqueeze(0)

        if attn_mask is not None:
            nW = attn_mask.shape[0]
            attn = attn.view(-1, nW, self.num_heads, N, N)
            attn = attn + attn_mask.unsqueeze(1)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


# ----------------------------
# Swin block 3D
# ----------------------------
class SwinBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size, mlp_ratio):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio)

    def build_attn_mask(self, D, H, W, device):
        sd, sh, sw = self.shift_size
        if sd == sh == sw == 0:
            return None
        wd, wh, ww = self.window_size
        img_mask = torch.zeros((1, D, H, W, 1), device=device)
        cnt = 0
        for ds in (slice(0, -wd), slice(-wd, -sd), slice(-sd, None)):
            for hs in (slice(0, -wh), slice(-wh, -sh), slice(-sh, None)):
                for ws in (slice(0, -ww), slice(-ww, -sw), slice(-sw, None)):
                    img_mask[:, ds, hs, ws, :] = cnt
                    cnt += 1
        mask_windows = window_partition_3d(img_mask, self.window_size).reshape(-1, wd * wh * ww)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0)

    def forward(self, x, D, H, W):
        B, L, C = x.shape
        wd, wh, ww = self.window_size
        sd, sh, sw = self.shift_size

        if L != D * H * W:
            raise RuntimeError(f"L mismatch: L={L} vs D*H*W={D*H*W} (D,H,W={D,H,W})")

        x0 = x
        x = self.norm1(x).reshape(B, D, H, W, C)

        # shift
        if sd or sh or sw:
            x = torch.roll(x, shifts=(-sd, -sh, -sw), dims=(1, 2, 3))

        # pad in token-space to multiples of window_size
        pad_d = (wd - (D % wd)) % wd
        pad_h = (wh - (H % wh)) % wh
        pad_w = (ww - (W % ww)) % ww
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        Dp, Hp, Wp = D + pad_d, H + pad_h, W + pad_w

        N = wd * wh * ww

        # padding attention mask: block attention to padded tokens
        valid = x.new_zeros((1, Dp, Hp, Wp, 1))
        valid[:, :D, :H, :W, :] = 1.0
        valid_w = window_partition_3d(valid, self.window_size).reshape(-1, N)  # (nW, N)
        pad_attn = (valid_w.unsqueeze(1) * valid_w.unsqueeze(2))               # (nW, N, N)
        pad_attn = pad_attn.masked_fill(pad_attn == 0, -100.0).masked_fill(pad_attn == 1, 0.0)

        # shifted-window mask on padded sizes
        shift_attn = self.build_attn_mask(Dp, Hp, Wp, x.device)  # (nW, N, N) or None
        attn_mask = pad_attn if shift_attn is None else (pad_attn + shift_attn)

        # window attention
        xw = window_partition_3d(x, self.window_size).reshape(-1, N, C)
        xw = self.attn(xw, attn_mask)
        x = window_reverse_3d(xw.reshape(-1, wd, wh, ww, C), self.window_size, Dp, Hp, Wp)

        # unpad back
        x = x[:, :D, :H, :W, :]

        # reverse shift
        if sd or sh or sw:
            x = torch.roll(x, shifts=(sd, sh, sw), dims=(1, 2, 3))

        x = x.reshape(B, L, C)
        x = x0 + x
        x = x + self.mlp(self.norm2(x))
        return x



# ----------------------------
# patch embed / merge / expand
# ----------------------------
class PatchEmbed3D(nn.Module):
    def __init__(self, in_chans, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        D, H, W = x.shape[-3:]
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), D, H, W


class PatchMerging3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(8 * dim)
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)

    def forward(self, x, D, H, W):
        B, _, C = x.shape
        x = x.view(B, D, H, W, C)
        xs = [x[:, d::2, h::2, w::2, :] for d in (0, 1) for h in (0, 1) for w in (0, 1)]
        x = torch.cat(xs, dim=-1)
        D, H, W = D // 2, H // 2, W // 2
        x = x.view(B, D * H * W, -1)
        return self.reduction(self.norm(x)), D, H, W


class PatchExpanding3D(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.expand = nn.Linear(in_dim, 8 * out_dim)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x, D, H, W):
        B, _, C = x.shape
        x = self.expand(self.norm(x)).view(B, D, H, W, 2, 2, 2, -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        D, H, W = 2 * D, 2 * H, 2 * W
        return x.view(B, D * H * W, -1), D, H, W


# ----------------------------
# block masking
# ----------------------------
def block_mask_3d(B, D, H, W, block, ratio, device):
    bd, bh, bw = block
    total = D * H * W
    target = int(total * ratio)
    mask = torch.zeros((B, D, H, W), device=device, dtype=torch.bool)
    masked = torch.zeros(B, device=device)
    while (masked < target).any():
        d0 = torch.randint(0, max(1, D - bd + 1), (B,), device=device)
        h0 = torch.randint(0, max(1, H - bh + 1), (B,), device=device)
        w0 = torch.randint(0, max(1, W - bw + 1), (B,), device=device)
        for b in range(B):
            if masked[b] >= target:
                continue
            region = mask[b, d0[b]:d0[b]+bd, h0[b]:h0[b]+bh, w0[b]:w0[b]+bw]
            new = (~region).sum()
            mask[b, d0[b]:d0[b]+bd, h0[b]:h0[b]+bh, w0[b]:w0[b]+bw] = True
            masked[b] += new
    return mask.view(B, -1).float()


# ----------------------------
# patchify target
# ----------------------------


# ----------------------------
# config + model
# ----------------------------
def unpatchify_3d(x_patches, patch, in_chans, grid_size_dhw):
    """
    x_patches: (B, N, pD*pH*pW*C)
    patch: (pD,pH,pW)
    grid_size_dhw: (Dg,Hg,Wg) tokens grid
    returns: (B, C, Dg*pD, Hg*pH, Wg*pW)
    """
    pD, pH, pW = patch
    Dg, Hg, Wg = grid_size_dhw
    B, N, pv = x_patches.shape
    C = in_chans

    # (B, Dg, Hg, Wg, C, pD, pH, pW)
    x = x_patches.reshape(B, Dg, Hg, Wg, C, pD, pH, pW)
    # (B, C, Dg, pD, Hg, pH, Wg, pW)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    # (B, C, D, H, W)
    return x.reshape(B, C, Dg * pD, Hg * pH, Wg * pW)


class SwinMAE3D(nn.Module):
    def __init__(self, in_chans: int = 1, patch_size: Tuple[int,int,int] = (2,4,4), embed_dim: int = 96, depths: Tuple[int,...] = (2,2,2,2) , 
    num_heads: Tuple[int,...] = (3,6,12,24) ,window_size: Tuple[int,int,int] = (2,7,7) , mlp_ratio: float = 4.0 , mask_ratio: float = 0.75 ,
    block_size_tokens: Tuple[int,int,int] = (2,4,4) ):
        super().__init__()
        self.in_chans = in_chans
        self.patch_size = _ntuple3(patch_size)
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = _ntuple3(window_size)
        self.mlp_ratio = mlp_ratio
        self.mask_ratio = mask_ratio
        self.block_size_tokens = _ntuple3(block_size_tokens)
   
        self.patch_embed = PatchEmbed3D(self.in_chans, self.embed_dim, self.patch_size)
        self.mask_token = nn.Parameter(torch.zeros(1,1,self.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        dims = [self.embed_dim * (2**i) for i in range(len(self.depths))]
        self.enc = nn.ModuleList()
        for i,(d,h) in enumerate(zip(self.depths, self.num_heads)):
            blocks = nn.ModuleList([
                SwinBlock3D(dims[i], h, self.window_size,
                            (0,0,0) if j%2==0 else tuple(w//2 for w in self.window_size),
                            self.mlp_ratio)
                for j in range(d)
            ])
            merge = PatchMerging3D(dims[i]) if i < len(self.depths)-1 else None
            self.enc.append(nn.ModuleDict({"blocks": blocks, "merge": merge}))
        self.enc_norm = nn.LayerNorm(dims[-1])

        dec_dims = list(reversed(dims))
        self.dec = nn.ModuleList()
        for i,(d,h) in enumerate(zip(reversed(self.depths), reversed(self.num_heads))):
            blocks = nn.ModuleList([
                SwinBlock3D(dec_dims[i], h, self.window_size,
                            (0,0,0) if j%2==0 else tuple(w//2 for w in self.window_size),
                            self.mlp_ratio)
                for j in range(d)
            ])
            expand = PatchExpanding3D(dec_dims[i], dec_dims[i+1]) if i < len(dec_dims)-1 else None
            self.dec.append(nn.ModuleDict({"blocks": blocks, "expand": expand}))

        self.dec_norm = nn.LayerNorm(dec_dims[-1])
        pd,ph,pw =self.patch_size
        self.head = nn.Linear(dec_dims[-1], pd*ph*pw*self.in_chans)

    def forward(self, x):
        """
        Returns:
        recon: (B, C, D, H, W)  reconstructed volume, same shape as input x
        Assumes x is already padded/cropped so (D,H,W) divisible by patch_size.
        """
        B, C, Dv, Hv, Wv = x.shape

        # patch embed -> tokens on patch grid
        t, D, H, W = self.patch_embed(x)  # t: (B, N, embed_dim), N=D*H*W

        # MAE token masking (still kept to preserve MAE behavior)
        mask = block_mask_3d(B, D, H, W, self.block_size_tokens, self.mask_ratio, x.device)
        t = t * (1 - mask.unsqueeze(-1)) + self.mask_token * mask.unsqueeze(-1)

        # encoder
        for s in self.enc:
            for blk in s["blocks"]:
                t = blk(t, D, H, W)
            if s["merge"] is not None:
                t, D, H, W = s["merge"](t, D, H, W)
        t = self.enc_norm(t)

        # decoder
        for s in self.dec:
            for blk in s["blocks"]:
                t = blk(t, D, H, W)
            if s["expand"] is not None:
                t, D, H, W = s["expand"](t, D, H, W)

        # token -> patch voxels
        t = self.dec_norm(t)
        pred = self.head(t)  # (B, N, pD*pH*pW*C)

        # unpatchify to volume so trainer can do criterion(pred, x)
        recon = unpatchify_3d(pred, self.patch_size, C, (D, H, W))  # (B, C, Dv, Hv, Wv)
        return recon




if __name__ == "__main__":
    import torch

    # --- paramètres (équivalent JSON chargé) ---
    params = {
        "in_chans": 1,
        "patch_size": (2, 4, 4),
        "embed_dim": 96,
        "depths": (2, 2, 2, 2),
        "num_heads": (3, 6, 12, 24),
        "window_size": (2, 7, 7),
        "mlp_ratio": 4.0,
        "mask_ratio": 0.75,
        "block_size_tokens": (2, 4, 4),
    }

    # --- init modèle ---
    model = SwinMAE3D(**params)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # --- input dummy ---
    # Contraintes :
    # - divisible par patch_size
    # - après patch embed, divisible par window_size
    # - divisible par 2**(len(depths)-1)
    x = torch.randn(1, 1, 32, 224, 224, device=device)

    # --- forward ---
    with torch.no_grad():
        out = model(x)

    print("Model initialized successfully")
    print("pred shape:", out["pred"].shape)
    print("mask shape:", out["mask"].shape)
    print("loss:", out["loss"].item())
