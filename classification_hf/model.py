"""
Verbatim copy of curia/modeling_dinov2.py attention classes + curia/trainer.py Classifier.
No modifications — the goal is an exact replica of the curia pipeline.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Attention classes (verbatim from curia/modeling_dinov2.py) ────────────────


@dataclass
class AttentionConfig:
    num_heads: int
    num_queries: int
    use_norm: bool = True
    use_skip_connection: bool = True
    attention_block: List[str] = field(default_factory=lambda: ["self", "cross"])


class Attention(nn.Module):
    def __init__(self, out_dim: int, use_skip_connection: bool = True,
                 use_norm: bool = True, num_heads: int = 1):
        super().__init__()
        self.out_dim = out_dim
        self.multihead_attn = nn.MultiheadAttention(out_dim, num_heads, batch_first=True)
        self.use_norm = use_norm
        self.use_skip_connection = use_skip_connection
        if self.use_norm:
            self.norm = nn.LayerNorm(out_dim)

    def forward(self, query, key, value, mask_attention=None):
        attn_output, attn_output_weights = self.multihead_attn(
            query, key, value, attn_mask=mask_attention
        )
        if self.use_skip_connection:
            attn_output = query + attn_output
        if self.use_norm:
            attn_output = self.norm(attn_output)
        return attn_output, attn_output_weights


class SelfAttention(nn.Module):
    def __init__(self, out_dim: int, use_skip_connection: bool = True,
                 use_norm: bool = True, num_heads: int = 1):
        super().__init__()
        self.attention = Attention(out_dim, use_skip_connection, use_norm, num_heads)

    def forward(self, feature: torch.Tensor, mask_attention: Optional[torch.Tensor] = None):
        if mask_attention is not None:
            attn_mask  = mask_attention.unsqueeze(1) & mask_attention.unsqueeze(2)
            attn_mask2 = (~mask_attention.unsqueeze(1)) & (~mask_attention.unsqueeze(2))
            mask_attention = ~(attn_mask + attn_mask2)
        return self.attention(feature, feature, feature, mask_attention)


class CrossAttention(nn.Module):
    def __init__(self, out_dim: int, use_skip_connection: bool = True,
                 use_norm: bool = True, num_heads: int = 1, num_queries: int = 1):
        super().__init__()
        self.attention = Attention(out_dim, use_skip_connection, use_norm, num_heads)
        self.num_queries = num_queries
        self.learned_queries = nn.Parameter(torch.randn(num_queries, out_dim))

    def forward(self, feature: torch.Tensor, mask_attention: Optional[torch.Tensor] = None):
        B = feature.size(0)
        learned_queries = self.learned_queries.unsqueeze(0).repeat(B, 1, 1)
        if mask_attention is not None:
            mask_attention = ~mask_attention.unsqueeze(1).expand(-1, self.num_queries, -1)
        return self.attention(learned_queries, feature, feature, mask_attention)


class AttentionModule(nn.Module):
    def __init__(self, config: AttentionConfig, out_dim: int):
        super().__init__()
        self.attention_block = config.attention_block
        if "self" in self.attention_block:
            self.self_attention = SelfAttention(
                out_dim, num_heads=config.num_heads,
                use_norm=config.use_norm, use_skip_connection=config.use_skip_connection,
            )
        if "cross" in self.attention_block:
            self.cross_attention = CrossAttention(
                out_dim, num_heads=config.num_heads, num_queries=config.num_queries,
                use_norm=config.use_norm, use_skip_connection=config.use_skip_connection,
            )

    def forward(self, x: torch.Tensor):
        attention_weights_list = []
        for block in self.attention_block:
            mask_attention = (x != 0).any(dim=-1)
            if block == "self":
                x, attention_weights = self.self_attention(x, mask_attention)
            elif block == "cross":
                x, attention_weights = self.cross_attention(x, mask_attention)
            else:
                raise ValueError(f"Unknown attention block {block}")
            attention_weights_list.append(attention_weights)
        x = x.mean(dim=1)
        return x, attention_weights_list


# ── Classifier (verbatim from curia/trainer.py) ───────────────────────────────


class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim, regression=False, attention_cfg=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.regression = regression

        self.attention_module = None
        if attention_cfg:
            attn_config = AttentionConfig(
                num_heads=attention_cfg.get("num_heads", 1),
                num_queries=attention_cfg.get("num_queries", 1),
                use_norm=True,
                use_skip_connection=True,
                attention_block=list(attention_cfg.get("block", ("self", "cross"))),
            )
            self.attention_module = AttentionModule(attn_config, in_dim)

        self.linear = nn.Linear(in_dim, out_dim)
        self.loss_fn = nn.MSELoss() if regression else nn.CrossEntropyLoss()

    def forward(self, pixel_values, labels=None):
        if self.attention_module:
            features, _ = self.attention_module(pixel_values)
        else:
            features = pixel_values

        logits = self.linear(features)
        if self.regression:
            logits = logits.squeeze(-1)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return {"logits": logits, "loss": loss}


# ── TokenGridClassifier (CNN sur la grille spatiale de tokens DINOv2) ─────────


class BasicBlock(nn.Module):
    """Standard residual block: two 3×3 convolutions with BN and skip connection."""
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))


class TokenGridClassifier(nn.Module):
    """
    Classifies DINOv2 patch tokens by treating them as a 2D spatial feature map.

    Input : pixel_values — (B, N, D) with N = grid² (e.g. 1024 = 32×32)
    Output: {"logits": (B, num_classes), "loss": scalar or None}

    Architecture
        reshape  : (B, N, D)  →  (B, D, grid, grid)
        proj     : Conv1×1 D → proj_dim  + BN + ReLU
        blocks   : n_blocks × BasicBlock(proj_dim)
        pool     : AdaptiveAvgPool2d(1)  →  (B, proj_dim)
        fc       : Linear(proj_dim, num_classes)
    """

    def __init__(self, in_dim: int, num_classes: int,
                 proj_dim: int = 128, n_blocks: int = 2,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, proj_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.ReLU(inplace=True),
        )
        self.blocks  = nn.Sequential(*[BasicBlock(proj_dim) for _ in range(n_blocks)])
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Linear(proj_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, pixel_values: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> dict:
        B, N, D = pixel_values.shape
        grid = int(N ** 0.5)
        x = pixel_values.permute(0, 2, 1).reshape(B, D, grid, grid)
        x = self.proj(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)   # (B, proj_dim)
        logits = self.fc(x)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return {"logits": logits, "loss": loss}


# ── MaskedBackboneClassifier (unfrozen backbone + masked avg pool) ────────────


class MaskedBackboneClassifier(nn.Module):
    """
    DINOv2 backbone (unfrozen or frozen) + masked-average-pool + linear head.
    Replicates the curia inference pipeline end-to-end so that gradients can
    flow through the backbone.

    forward(pixel_values, mask=None, labels=None) → {"logits": ..., "loss": ...}
    mask : (B, 1, crop_size, crop_size) binary float — same format as preprocess_function
    """

    def __init__(self, backbone, hidden_size: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.linear   = nn.Linear(hidden_size, num_classes)
        self.loss_fn  = nn.CrossEntropyLoss()

    def forward(self, pixel_values: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> dict:
        tokens = self.backbone(pixel_values=pixel_values).last_hidden_state[:, 1:]  # (B, N, D)
        if mask is not None:
            B, N, D = tokens.shape
            grid       = int(N ** 0.5)
            patch_size = mask.shape[-1] // grid
            m = F.max_pool2d(mask.float(), kernel_size=patch_size, stride=patch_size)
            m = m.view(B, N, 1)
            features = (tokens * m).sum(1) / m.sum(1).clamp(min=1e-6)
        else:
            features = tokens.mean(1)
        logits = self.linear(features)
        loss   = self.loss_fn(logits, labels) if labels is not None else None
        return {"logits": logits, "loss": loss}
