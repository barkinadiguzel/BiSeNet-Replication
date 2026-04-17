import torch.nn.functional as F

def upsample(x, size):
    return F.interpolate(x, size=size, mode='bilinear', align_corners=True)
