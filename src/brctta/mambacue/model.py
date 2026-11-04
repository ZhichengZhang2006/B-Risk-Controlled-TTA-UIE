"""Stage Mamba + complementary priors + original color/detail feature heads."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .prior import ComplementaryPriorMaps, ComplementaryPriorFusion, normalize_prior_types
from .encoder import Encoder
from .output_heads import SpatialColorCorrection, HighFreqDetailInjector


class ConvBlock(nn.Module):
    """Standard two-convolution U-Net block."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        )
        self.block = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
        return self.block(torch.cat([x, skip], dim=1))


class Decoder(nn.Module):
    def __init__(self, base_ch=32, prior_types=("color", "illumination", "contrast"),
                 prior_width=16):
        super().__init__()
        self.dec3 = DecoderBlock(base_ch * 8, base_ch * 4, base_ch * 4)
        self.dec2 = DecoderBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.dec1 = DecoderBlock(base_ch * 2, base_ch, base_ch)
        self.prior3 = (ComplementaryPriorFusion(base_ch * 4, prior_types, prior_width)
                       if prior_types else None)
        self.prior2 = (ComplementaryPriorFusion(base_ch * 2, prior_types, prior_width)
                       if prior_types else None)

    def forward(self, e1, e2, e3, bottleneck, prior_maps=None):
        gates = {}
        d3 = self.dec3(bottleneck, e3)
        if self.prior3 is not None:
            d3, gates["h4"] = self.prior3(d3, prior_maps)
        d2 = self.dec2(d3, e2)
        if self.prior2 is not None:
            d2, gates["h2"] = self.prior2(d2, prior_maps)
        return self.dec1(d2, e1), gates


class myModel(nn.Module):
    def __init__(
        self, in_channels=3, hidden_channels=32, out_channels=3, lowres_size=32,
        prior_types=("color", "illumination", "contrast"), prior_width=16, prior_window=15,
        stage_depths=(1, 1, 1, 1), scan_mode="four", use_checkpoint=False,
        output_heads="both",
    ):
        super().__init__()
        # lowres_size is retained only for API/checkpoint-script compatibility.
        del lowres_size
        if in_channels != 3 or out_channels != 3:
            raise ValueError("This variant expects RGB input/output")
        if hidden_channels < 8 or hidden_channels % 8:
            raise ValueError("hidden_channels must be a positive multiple of 8")
        if output_heads not in ("none", "color", "detail", "both"):
            raise ValueError("output_heads must be none, color, detail or both")
        self.output_heads = output_heads
        self.prior_types = normalize_prior_types(prior_types)
        self.prior_builder = ComplementaryPriorMaps(window_size=prior_window)
        self.encoder = Encoder(in_channels, hidden_channels, stage_depths,
                               scan_mode, use_checkpoint)
        self.decoder = Decoder(hidden_channels, self.prior_types, prior_width)
        self.color_correction = (SpatialColorCorrection(hidden_channels)
                                 if output_heads in ("color", "both") else None)
        self.detail_inject = (HighFreqDetailInjector(hidden_channels)
                              if output_heads in ("detail", "both") else None)
        self.output_conv = nn.Conv2d(hidden_channels, out_channels, 3, padding=1)

    def build_prior(self, x):
        return self.prior_builder(x)

    def forward(self, x, prior_map=None, return_aux=False):
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 8:
            raise ValueError("Expected [B, 3, H, W] input with height and width >= 8")
        if self.prior_types:
            if prior_map is None:
                prior_map = self.build_prior(x)
            elif (prior_map.ndim != 4 or prior_map.shape != (x.shape[0], 4, *x.shape[-2:])
                  or not prior_map.is_floating_point()):
                raise ValueError("prior_map must be floating-point [B, 4, H, W], ordered cr/cb/L/C")
            prior_map = prior_map.to(device=x.device, dtype=x.dtype)
        else:
            prior_map = None
        e1, e2, e3, bottleneck = self.encoder(x)
        decoded, gates = self.decoder(e1, e2, e3, bottleneck, prior_map)
        decoded = self.color_correction(decoded, x)
        decoded = self.detail_inject(decoded)
        out = torch.sigmoid(self.output_conv(decoded))
        if return_aux:
            return out, {"prior_maps": prior_map, "prior_gates": gates}
        return out
