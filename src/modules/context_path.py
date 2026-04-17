import torch
import torch.nn as nn
import torchvision.models as models
from src.blocks.arm import AttentionRefinementModule
from src.blocks.utils import upsample

class ContextPath(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.resnet18(pretrained=True)

        self.stage1 = nn.Sequential(*list(backbone.children())[:6])   
        self.stage2 = backbone.layer3 
        self.stage3 = backbone.layer4  

        self.arm16 = AttentionRefinementModule(256, 256)
        self.arm32 = AttentionRefinementModule(512, 512)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        feat8 = self.stage1(x)
        feat16 = self.stage2(feat8)
        feat32 = self.stage3(feat16)

        arm32 = self.arm32(feat32)
        gp = self.global_pool(feat32)
        gp = torch.nn.functional.interpolate(gp, size=arm32.shape[2:], mode='bilinear')

        feat32 = arm32 + gp
        feat32_up = upsample(feat32, feat16.shape[2:])

        arm16 = self.arm16(feat16)
        feat16 = arm16 + feat32_up

        return feat16, feat32
