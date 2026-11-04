# Original feature modulation modules, preserved for controlled ablations.
import torch
from torch import nn
from torch.nn import functional as F


class SpatialColorCorrection(nn.Module):
    def __init__(self, feat_ch):
        super().__init__()
        self.color_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, feat_ch * 2, 3, padding=1)
        )
        nn.init.zeros_(self.color_encoder[-1].weight)
        nn.init.zeros_(self.color_encoder[-1].bias)

    def forward(self, feat, rgb):
        # feat: [B, C, H, W], rgb: [B, 3, H, W]
        color_map = F.interpolate(rgb, size=feat.shape[2:], mode='bilinear', align_corners=False)
        gamma_beta = self.color_encoder(color_map)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        return feat * (1.0 + gamma) + beta


class HighFreqDetailInjector(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                 dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('laplacian', laplacian.repeat(in_ch, 1, 1, 1))
        self.detail_conv = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch // 4, 1), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 4, in_ch, 1), nn.Sigmoid(),
        )
        self.detail_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        hf = F.conv2d(x, self.laplacian, padding=1, groups=x.shape[1])
        weight = self.ca(x)
        detail = self.detail_conv(hf) * weight
        return x + self.detail_scale * detail
