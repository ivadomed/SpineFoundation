import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


class PatchWiseSegHead(nn.Module):
    """Simple 2-layer head: single bilinear ×N upsample (baseline)."""
    def __init__(self, in_channels: int, hidden_channels: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def _make_norm(norm_type: str, num_channels: int) -> nn.Module:
    if norm_type == "group":
        num_groups = 1
        for g in [8, 4, 2, 1]:
            if num_channels % g == 0:
                num_groups = g
                break
        return nn.GroupNorm(num_groups, num_channels)
    elif norm_type == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True)
    else:
        # Lower momentum (0.05 vs default 0.1) stabilises running stats
        # when few batches per epoch (small datasets).
        return nn.BatchNorm2d(num_channels, momentum=0.05)


def _up_block(in_ch: int, out_ch: int, dropout: float = 0.0, norm_type: str = "batch", nonlin: str = "gelu") -> nn.Sequential:
    # InstanceNorm (nnUNet-style) works better with conv_bias=True
    use_bias = norm_type == "instance"

    def _act() -> nn.Module:
        if nonlin == "leakyrelu":
            return nn.LeakyReLU(negative_slope=0.01, inplace=True)
        return nn.GELU()

    layers: list[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=use_bias),
        _make_norm(norm_type, out_ch),
        _act(),
    ]
    if dropout > 0.0:
        layers.append(nn.Dropout2d(dropout))
    layers += [
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=use_bias),
        _make_norm(norm_type, out_ch),
        _act(),
    ]
    return nn.Sequential(*layers)


class DeepSegHead(nn.Module):
    """Progressive ×2 upsampling decoder with configurable depth.

    depth=4 (default): 32→64→128→256→512  (exact match for patch_size=16, image_size=512)
    depth=3:           32→64→128→256       + bilinear ×2 in forward_from_tokens
    depth=2:           32→64→128           + bilinear ×4 in forward_from_tokens

    hidden_channels controls the base width (default 256, use 64 for small datasets).
    dropout: Dropout2d rate applied after first conv in each up block (0 = disabled).
    norm_type: "batch" (default), "group", or "instance" (nnUNet-style).
    nonlin: "gelu" (default) or "leakyrelu" (nnUNet-style).
    """
    def __init__(self, in_channels: int, hidden_channels: int = 256, dropout: float = 0.0,
                 norm_type: str = "batch", nonlin: str = "gelu", depth: int = 4):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        h = hidden_channels
        kw = dict(dropout=dropout, norm_type=norm_type, nonlin=nonlin)
        self.stem = _up_block(in_channels, h, **kw)
        self.ups  = nn.ModuleList()
        ch = h
        for _ in range(depth):
            self.ups.append(_up_block(ch, ch // 2, **kw))
            ch //= 2
        self.out = nn.Conv2d(ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for up in self.ups:
            x = up(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        return self.out(x)


class FrozenBackboneWithSegHead(nn.Module):
    def __init__(self, model_dir: str, seg_head_channels: int = 256, seg_head_dropout: float = 0.0,
                 seg_head_norm: str = "batch", seg_head_nonlin: str = "gelu", seg_head_depth: int = 4):
        super().__init__()

        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(model_dir, config=config, trust_remote_code=True)

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        if hasattr(self.backbone, "head") and isinstance(self.backbone.head, nn.Module):
            for p in self.backbone.head.parameters():
                p.requires_grad = False

        hidden_size = getattr(config, "hidden_size", None) or getattr(config, "embed_dim", None)
        if hidden_size is None:
            raise ValueError("Cannot infer hidden size from config. Need `hidden_size` or `embed_dim`.")

        self.patch_size = int(getattr(config, "patch_size", 14))
        self.seg_head = DeepSegHead(
            in_channels=hidden_size,
            hidden_channels=seg_head_channels,
            dropout=seg_head_dropout,
            norm_type=seg_head_norm,
            nonlin=seg_head_nonlin,
            depth=seg_head_depth,
        )

    def train(self, mode: bool = True):
        """Keep backbone permanently in eval mode regardless of the training mode flag."""
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def extract_patch_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        outputs = self.backbone(pixel_values=x, output_hidden_states=False, return_dict=True)
        if hasattr(outputs, "last_hidden_state"):
            tokens = outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            tokens = outputs[0]
        else:
            raise ValueError("Backbone output format not supported. Could not find patch tokens.")

        _, n, _ = tokens.shape
        h, w = x.shape[-2:]
        gh = h // self.patch_size
        gw = w // self.patch_size
        expected_patches = gh * gw

        if expected_patches <= 0:
            raise ValueError(f"Invalid input size for patch extraction: H={h}, W={w}, patch_size={self.patch_size}")

        if n == expected_patches:
            patch_tokens = tokens
            return patch_tokens, gh, gw

        if n > expected_patches:
            special = n - expected_patches
            patch_tokens = tokens[:, special:, :]
            return patch_tokens, gh, gw

        raise ValueError(
            f"Token count {n} is smaller than expected patch grid {expected_patches} "
            f"(H={h}, W={w}, patch_size={self.patch_size})."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_tokens, gh, gw = self.extract_patch_tokens(x)
        b, _, c = patch_tokens.shape

        feature_map = patch_tokens.transpose(1, 2).reshape(b, c, gh, gw)
        logits_patch = self.seg_head(feature_map)
        logits = F.interpolate(logits_patch, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits

    def forward_from_tokens(
        self,
        patch_tokens: torch.Tensor,
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        """Forward pass using pre-cached patch tokens — backbone is never called.

        Args:
            patch_tokens: (B, N, D) pre-extracted patch tokens (CLS already removed).
                          N must be a perfect square (square backbone input assumed).
            target_hw: (H, W) output resolution for bilinear upsampling.

        Returns:
            logits: (B, 1, H, W)
        """
        b, n, c = patch_tokens.shape
        gh = gw = int(n ** 0.5)
        if gh * gw != n:
            raise ValueError(
                f"Expected square patch grid but got {n} tokens (sqrt={n**0.5:.2f}). "
                "Only square inputs are supported for the cached-token fast path."
            )
        feature_map  = patch_tokens.transpose(1, 2).reshape(b, c, gh, gw)
        logits_patch = self.seg_head(feature_map)
        return F.interpolate(logits_patch, size=target_hw, mode="bilinear", align_corners=False)


class TrainableBackboneWithSegHead(nn.Module):
    """Same architecture as FrozenBackboneWithSegHead but backbone is trainable.

    Intended for end-to-end fine-tuning with differential learning rates:
      - backbone: small LR (e.g. lr * 0.1)
      - seg_head: full LR
    """

    def __init__(self, model_dir: str, seg_head_channels: int = 256, seg_head_dropout: float = 0.0,
                 seg_head_norm: str = "batch", seg_head_nonlin: str = "gelu", seg_head_depth: int = 4):
        super().__init__()

        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(model_dir, config=config, trust_remote_code=True)

        hidden_size = getattr(config, "hidden_size", None) or getattr(config, "embed_dim", None)
        if hidden_size is None:
            raise ValueError("Cannot infer hidden size from config.")

        self.patch_size = int(getattr(config, "patch_size", 14))
        self.seg_head = DeepSegHead(
            in_channels=hidden_size,
            hidden_channels=seg_head_channels,
            dropout=seg_head_dropout,
            norm_type=seg_head_norm,
            nonlin=seg_head_nonlin,
            depth=seg_head_depth,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=x, output_hidden_states=False, return_dict=True)
        if hasattr(outputs, "last_hidden_state"):
            tokens = outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)):
            tokens = outputs[0]
        else:
            raise ValueError("Backbone output format not supported.")

        _, n, _ = tokens.shape
        h, w = x.shape[-2:]
        gh = h // self.patch_size
        gw = w // self.patch_size
        expected = gh * gw

        patch_tokens = tokens[:, n - expected:, :] if n > expected else tokens

        b, _, c = patch_tokens.shape
        feature_map = patch_tokens.transpose(1, 2).reshape(b, c, gh, gw)
        logits_patch = self.seg_head(feature_map)
        return F.interpolate(logits_patch, size=x.shape[-2:], mode="bilinear", align_corners=False)
