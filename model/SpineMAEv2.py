"""
This script contains the SpineMAE model classes with 3D sin/cos positional embeddings (no learned absolute pos-emb).

Author: Thomas Dagonneau & Julien Laborde-Peyré (modified)
"""
import math
import torch
import torch.nn as nn
from monai.networks.blocks.transformerblock import TransformerBlock
import torch.nn.functional as F
from collections import OrderedDict

class PosEmbedCache:
    def __init__(self, max_items=32):
        self.max_items = max_items
        self._cache = OrderedDict()

    def get_3d(self, embed_dim, grid_size_dhw, device, dtype):
        gD, gH, gW = grid_size_dhw
        key = (embed_dim, gD, gH, gW, str(device), str(dtype))

        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        pe = get_3d_sincos_pos_embed(embed_dim, (gD, gH, gW), device=device).to(dtype=dtype)
        self._cache[key] = pe

        if len(self._cache) > self.max_items:
            self._cache.popitem(last=False)  # evict LRU
        return pe


class DynamicPatchEmbed3D(nn.Module):
    """
    Dynamic 3D patch embedding:
      input:  (B, in_channels, D, H, W)
      output: (B, N, embed_dim) with N = (D'//pD)*(H'//pH)*(W'//pW)
    """
    def __init__(self, in_channels=1, embed_dim=768, patch_size=(16,16,16), mode="pad", norm="layer"):
        super().__init__()
        self.patch_size = patch_size
        self.mode = mode  # "pad" or "crop"
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.norm = nn.LayerNorm(embed_dim) if norm == "layer" else None

    def _to_multiple(self, x):
        pD, pH, pW = self.patch_size
        B, C, D, H, W = x.shape
        if self.mode == "crop":
            D2 = (D // pD) * pD
            H2 = (H // pH) * pH
            W2 = (W // pW) * pW
            return x[..., :D2, :H2, :W2]
        if self.mode == "pad":
            D2 = ((D + pD - 1) // pD) * pD
            H2 = ((H + pH - 1) // pH) * pH
            W2 = ((W + pW - 1) // pW) * pW
            pad_d = D2 - D
            pad_h = H2 - H
            pad_w = W2 - W
            # F.pad order for 3D: (W_left,W_right,H_left,H_right,D_left,D_right)
            return F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d), mode="constant", value=0.0)
        raise ValueError("mode must be 'pad' or 'crop'")

    def forward(self, x):
        #x = self._to_multiple(x)
        x = self.proj(x)  # (B, embed_dim, gD, gH, gW)
        gD, gH, gW = x.shape[-3], x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim), N=gD*gH*gW
        if self.norm is not None:
            x = self.norm(x)
        return x, (gD, gH, gW)

def random_masking(x, mask_ratio):
    B, N, C = x.shape
    num_keep = int(N * (1 - mask_ratio))
    bruit = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(bruit, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :num_keep]
    x_visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, C))
    return x_visible, ids_restore


def _get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: int (must be even)
    pos: (M,) tensor in [0,1]
    return: (M, embed_dim)
    """
    assert embed_dim % 2 == 0
    half = embed_dim // 2
    omega = torch.arange(half, device=pos.device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / half))
    out = pos[:, None] * omega[None, :]
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    return emb


def get_3d_sincos_pos_embed(embed_dim, grid_size_dhw, device):
    """
    embed_dim: int, recommended divisible by 6
    grid_size_dhw: (gD, gH, gW)
    return: (N, embed_dim) where N=gD*gH*gW
    """
    gD, gH, gW = grid_size_dhw
    # normalize coords to [0,1]
    z = torch.linspace(0.0, 1.0, steps=gD, device=device)
    y = torch.linspace(0.0, 1.0, steps=gH, device=device)
    x = torch.linspace(0.0, 1.0, steps=gW, device=device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    pos_z = zz.reshape(-1)
    pos_y = yy.reshape(-1)
    pos_x = xx.reshape(-1)

    # split dims across axes
    # need each axis dim even for sin/cos; we allocate as evenly as possible and pad/truncate
    base = embed_dim // 3
    if base % 2 == 1:
        base -= 1
    dz = dy = dx = base
    used = dz + dy + dx
    if used == 0:
        raise ValueError("embed_dim too small for 3D sin/cos pos embed. Use >= 6.")

    emb_z = _get_1d_sincos_pos_embed_from_grid(dz, pos_z)
    emb_y = _get_1d_sincos_pos_embed_from_grid(dy, pos_y)
    emb_x = _get_1d_sincos_pos_embed_from_grid(dx, pos_x)
    emb = torch.cat([emb_z, emb_y, emb_x], dim=1)  # (N, used)

    if used < embed_dim:
        pad = torch.zeros(emb.shape[0], embed_dim - used, device=device, dtype=emb.dtype)
        emb = torch.cat([emb, pad], dim=1)
    elif used > embed_dim:
        emb = emb[:, :embed_dim]
    return emb


class SpineEncoder(nn.Module):
    def __init__(self, in_channels=1, patch_size=(16, 16, 16),enc_embed_dim=120, enc_num_heads=12, enc_layers=12, enc_mlp_dim=3072, dropout=0, mask_ratio=0,pos_cache=None):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.enc_embed_dim = enc_embed_dim
        self.pos_cache = pos_cache

        self.patch_embed = DynamicPatchEmbed3D(
            in_channels=in_channels,
            embed_dim=enc_embed_dim,
            patch_size=patch_size,
            mode="pad"  # ou pad
        )
        

        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_size=enc_embed_dim, mlp_dim=enc_mlp_dim, num_heads=enc_num_heads, dropout_rate=dropout)
            for _ in range(enc_layers)
        ])

        self.norm = nn.LayerNorm(enc_embed_dim)
        
    def forward(self, x):
        tokens, (gD, gH, gW) = self.patch_embed(x)  # (B, N, C)
        # add sin/cos positional embedding
        if self.pos_cache is None:
            pos = get_3d_sincos_pos_embed(self.enc_embed_dim, (gD, gH, gW), device=tokens.device).to(dtype=tokens.dtype)
        else:
            pos = self.pos_cache.get_3d(self.enc_embed_dim, (gD, gH, gW), tokens.device, tokens.dtype)
        x = tokens + pos.unsqueeze(0)

        x_visible, ids_restore = random_masking(x, self.mask_ratio)
        z = x_visible

        for blk in self.transformer_layers:
            z = blk(z)
        z = self.norm(z)
        return z, ids_restore, (gD, gH, gW)


class SpineDecoder(nn.Module):
    def __init__(self,patch_size=(16, 16, 16), enc_embed_dim=256, dec_embed_dim=128, dec_layers=4, dec_num_heads=4, in_channels=1, dec_mlp_dim=3072, pos_cache=None):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.dec_embed_dim = dec_embed_dim


        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_embed_dim))


        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size=dec_embed_dim, mlp_dim=dec_mlp_dim, num_heads=dec_num_heads)
            for _ in range(dec_layers)
        ])
        self.norm = nn.LayerNorm(dec_embed_dim)

        pD, pH, pW = patch_size
        patch_voxels = pD * pH * pW * in_channels
        self.pred = nn.Linear(dec_embed_dim, patch_voxels)

        nn.init.normal_(self.mask_token, std=0.02)

    def _add_pos(self, x, grid_size):
        if self.pos_cache is None:
            pos = get_3d_sincos_pos_embed(self.dec_embed_dim, grid_size, device=x.device).to(dtype=x.dtype)
        else:
            pos = self.pos_cache.get_3d(self.dec_embed_dim, grid_size, x.device, x.dtype)
        return x + pos.unsqueeze(0)

    def embed(self, x_visible, ids_restore, grid_size):
        B, N_vis, _ = x_visible.shape
        N = ids_restore.shape[1]
        x = self.decoder_embed(x_visible)
        
        N_mask = N - N_vis
        mask_tokens = self.mask_token.expand(B, N_mask, -1)
        x = torch.cat([x, mask_tokens], dim=1)

        x = torch.gather(x, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        x = self._add_pos(x, grid_size)
        return x

    def unpatchify(self, x_patches, grid_size):
        B, N, pv = x_patches.shape
        pD, pH, pW = self.patch_size
        C = self.in_channels
        Dp, Hp, Wp = grid_size
        x = x_patches.view(B, Dp, Hp, Wp, C, pD, pH, pW)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        x = x.view(B, C, Dp * pD, Hp * pH, Wp * pW)
        return x

    def forward(self, x_visible, ids_restore, grid_size):
        x = self.embed(x_visible, ids_restore, grid_size)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        pred = self.pred(x)  # (B, N, patch_voxels)
        recon = self.unpatchify(pred, grid_size)  # (B, in_channels, D, H, W)
        return recon


class SpineMAE(nn.Module):
    def __init__(self, in_channels=1, patch_size=(16, 16, 16),
                 enc_embed_dim=120, enc_num_heads=12, enc_layers=12, enc_mlp_dim=3072, dropout=0, mask_ratio=0,
                 dec_embed_dim=128, dec_layers=4, dec_num_heads=4, dec_mlp_dim=3072):
        super().__init__()
        pos_cache = PosEmbedCache(max_items=32)
        self.encoder = SpineEncoder(
            in_channels=in_channels, patch_size=patch_size,
            enc_embed_dim=enc_embed_dim, enc_num_heads=enc_num_heads, enc_layers=enc_layers,
            enc_mlp_dim=enc_mlp_dim, dropout=dropout, mask_ratio=mask_ratio,pos_cache=pos_cache
        )
        self.decoder = SpineDecoder(
            patch_size=patch_size, enc_embed_dim=enc_embed_dim,
            dec_embed_dim=dec_embed_dim, dec_layers=dec_layers, dec_num_heads=dec_num_heads,
            in_channels=in_channels, dec_mlp_dim=dec_mlp_dim, pos_cache=pos_cache
        )

    def forward(self, x):
        z, ids_restore, grid_size = self.encoder(x)
        return self.decoder(z, ids_restore, grid_size)

