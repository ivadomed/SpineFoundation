import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


class PatchWiseSegHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class FrozenBackboneWithSegHead(nn.Module):
    def __init__(self, model_dir: str):
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

        self.seg_head = PatchWiseSegHead(in_channels=hidden_size)

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
