import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class ClassificationHead(nn.Module):
    """Lightweight MLP trained on top of frozen backbone features."""

    def __init__(self, in_features: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class FrozenBackboneForExtraction(nn.Module):
    """Frozen backbone used only for feature extraction (no gradient computation)."""

    def __init__(self, model_dir: str):
        super().__init__()
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(model_dir, config=config, trust_remote_code=True)

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        self.hidden_size = getattr(config, "hidden_size", None) or getattr(config, "embed_dim", None)
        if self.hidden_size is None:
            raise ValueError("Cannot infer hidden size from config. Need `hidden_size` or `embed_dim`.")

        self.patch_size = int(getattr(config, "patch_size", 14))

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the CLS token for each image: shape (B, hidden_size)."""
        outputs = self.backbone(pixel_values=x, output_hidden_states=False, return_dict=True)

        if hasattr(outputs, "last_hidden_state"):
            tokens = outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            tokens = outputs[0]
        else:
            raise ValueError("Unknown backbone output format.")

        # tokens: (B, 1 + n_patches, hidden_size)
        # Position 0 is the CLS token — best global representation for classification.
        return tokens[:, 0, :]
