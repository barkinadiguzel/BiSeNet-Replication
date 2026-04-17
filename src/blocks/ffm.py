import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.convblk = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, sp, cp):
        x = torch.cat([sp, cp], dim=1)
        x = self.convblk(x)
        attn = self.attention(x)
        return x + x * attn
