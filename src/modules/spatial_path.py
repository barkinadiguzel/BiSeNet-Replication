import torch.nn as nn
from src.blocks.conv import ConvBNReLU

class SpatialPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBNReLU(3, 64, stride=2)
        self.conv2 = ConvBNReLU(64, 128, stride=2)
        self.conv3 = ConvBNReLU(128, 256, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
