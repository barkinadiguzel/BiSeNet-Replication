import torch.nn as nn
from src.modules.spatial_path import SpatialPath
from src.modules.context_path import ContextPath
from src.blocks.ffm import FeatureFusionModule
from src.blocks.utils import upsample

class BiSeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.sp = SpatialPath()
        self.cp = ContextPath()

        self.ffm = FeatureFusionModule(256 + 256, 256)

        self.head = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        sp = self.sp(x)
        cp16, _ = self.cp(x)

        cp16 = upsample(cp16, sp.shape[2:])

        feat = self.ffm(sp, cp16)
        out = self.head(feat)

        out = upsample(out, x.shape[2:])
        return out
