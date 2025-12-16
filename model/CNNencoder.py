"""
This script contains a resolution adaptive CNN feature extractor.

Author: Thomas Dagonneau & Julien Laborde-Peyré
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, padding=1, bias=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CNN3DFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, base_channels=32,feature_map_size=(2, 10, 10), num_blocks=4, adaptive_pool='max'):
        super().__init__()
        
        self.feature_map_size = feature_map_size
        self.num_blocks = int(num_blocks)

        if adaptive_pool not in ('avg', 'max'):
            raise ValueError("adaptive_pool must be 'avg' or 'max'")
        
        self.adaptive_pool = adaptive_pool

        channels = [in_channels] + [int(base_channels * (2 ** i)) for i in range(self.num_blocks)]
        blocks = []
        for i in range(self.num_blocks):
            blocks.append(ConvBlock3D(channels[i], channels[i + 1]))

        self.backbone = nn.Sequential(*blocks)
        self.out_channels = channels[-1]

    def forward(self, x, return_tokens=True):
        if any(x.shape[2:][d]<self.feature_map_size[d] for d in range(3)):
            raise ValueError("image trop petite")
        current = x
        base_size = tuple(int(s) for s in x.shape[2:])
        for i, block in enumerate(self.backbone):
            current = block(current)
            alpha = float(i + 1) / float(self.num_blocks)
            cur_spatial = current.shape[2:]
            desired = []
            for d in range(3):
                target_val = int(round(base_size[d] + alpha * (self.feature_map_size[d] - base_size[d])))              
                desired.append(target_val)
            if self.adaptive_pool == 'avg':
                current = F.adaptive_avg_pool3d(current, tuple(desired))
            else:
                current = F.adaptive_max_pool3d(current, tuple(desired))

        feat = current
        if not return_tokens:
            return feat
        B, C, D, H, W = feat.shape
        N = D * H * W
        tokens = feat.view(B, C, N).transpose(1, 2)
        return feat, tokens


if __name__ == '__main__':
    device = torch.device("cuda")
    model = CNN3DFeatureExtractor(in_channels=1, base_channels=16, num_blocks=4, feature_map_size=(2, 16, 16), adaptive_pool='max')
    summary(model, (16,1, 56, 512, 512),depth=20)















    "Pour afficher l'évolution des résolutions et des channels : "
    plot_res = True
    if plot_res:
        import numpy as np
        import matplotlib.pyplot as plt

        feature_map_size = tuple(int(s) for s in model.feature_map_size)
        num_blocks = model.num_blocks

        initial_sizes = [
            (8, 64, 64),
            (16, 128, 128),
            (32, 256, 256),
            (56, 512, 512)]

        axes = ['D', 'H', 'W']
        # Add a 4th subplot for channels
        fig, axs = plt.subplots(1, 4, figsize=(20, 4))

        for init in initial_sizes:
            base_size = tuple(int(s) for s in init)
            sizes_per_axis = {0: [], 1: [], 2: []}

            for i in range(num_blocks):
                alpha = float(i + 1) / float(num_blocks)
                for d in range(3):
                    target_val = int(round(base_size[d] + alpha * (feature_map_size[d] - base_size[d])))
                    sizes_per_axis[d].append(target_val)

            x = list(range(1, num_blocks + 1))
            for d in range(3):
                axs[d].plot(x, sizes_per_axis[d], marker='o', label=f'init={init}')
                axs[d].set_xlabel('block')
                axs[d].set_title(f'Axis {axes[d]}')
                axs[d].grid(True)

        # Legend for spatial axes
        for ax in axs[:3]:
            ax.legend()

        # Channels per block (same across initial sizes) - compute from model
        channel_counts = [getattr(block.conv, 'out_channels', None) for block in model.backbone]
        if any(c is None for c in channel_counts):
            # fallback: try reading conv.in_channels of first block and doubling
            base_ch = getattr(model.backbone[0].conv, 'in_channels', 1)
            channel_counts = [int(base_ch * (2 ** i)) for i in range(num_blocks)]

        axs[3].plot(x, channel_counts, marker='s', color='k', label='channels')
        axs[3].set_xlabel('block')
        axs[3].set_title('Channels')
        axs[3].grid(True)
        axs[3].legend()
        plt.tight_layout()

        import os
        out_dir = os.path.join(os.path.dirname(__file__), 'figures')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'resolution_evolution.png')
        plt.savefig(out_path, dpi=150)
        print(f'Saved resolution evolution figure to: {out_path}')

    