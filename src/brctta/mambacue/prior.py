"""Complementary RGB-derived cues and independently gated decoder residuals.

The maps are image statistics, not estimates of physical transmission or depth.
Channel order: relative red, relative blue, low-frequency luminance, contrast.
"""
import torch
from torch import nn
from torch.nn import functional as F


PRIOR_CHANNELS = {"color": (0, 2), "illumination": (2, 3), "contrast": (3, 4)}


def normalize_prior_types(prior_types):
    names = tuple(prior_types)
    if names == ("none",) or not names:
        return ()
    if len(set(names)) != len(names) or any(name not in PRIOR_CHANNELS for name in names):
        raise ValueError("Use unique color/illumination/contrast names, or 'none' alone")
    return tuple(name for name in PRIOR_CHANNELS if name in names)


def add_prior_arguments(parser):
    parser.add_argument(
        "--prior_types", nargs="+", default=list(PRIOR_CHANNELS),
        choices=[*PRIOR_CHANNELS, "none"],
        help="Enabled independent prior branches; use 'none' to disable prior fusion",
    )


class ComplementaryPriorMaps(nn.Module):
    def __init__(self, window_size=15, eps=1e-3):
        super().__init__()
        if not isinstance(window_size, int) or window_size < 3 or window_size % 2 == 0:
            raise ValueError("window_size must be an odd integer >= 3")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.window_size = window_size
        self.eps = eps

    def _average(self, x):
        padding = self.window_size // 2
        # Replication preserves constant images at borders and supports small images.
        return F.avg_pool2d(F.pad(x, (padding,) * 4, mode="replicate"),
                            self.window_size, stride=1)

    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1] != 3 or not rgb.is_floating_point():
            raise ValueError("Expected floating-point RGB input [B, 3, H, W] in [0, 1]")
        # Compute ratios and local moments in FP32 when the model uses reduced precision.
        work = rgb.float() if rgb.dtype in (torch.float16, torch.bfloat16) else rgb
        smooth_rgb = self._average(work)
        rgb_sum = smooth_rgb.sum(dim=1, keepdim=True)
        red = smooth_rgb[:, 0:1] / (rgb_sum + self.eps)
        blue = smooth_rgb[:, 2:3] / (rgb_sum + self.eps)

        luminance = 0.299 * work[:, 0:1] + 0.587 * work[:, 1:2] + 0.114 * work[:, 2:3]
        # Center moments to avoid cancellation in E[Y^2]-E[Y]^2 on uniform images.
        # A constant shift leaves local variance unchanged; detach avoids a needless
        # global gradient path through that algebraically cancelling shift.
        center = luminance.mean(dim=(2, 3), keepdim=True).detach()
        centered = luminance - center
        local_centered = self._average(centered)
        illumination = local_centered + center
        variance = (self._average(centered.square()) - local_centered.square()).clamp_min(0)
        # Smoothed sqrt: zero at variance=0, with a finite derivative in flat/dark areas.
        sigma = (variance + self.eps ** 2).sqrt() - self.eps
        contrast = sigma / (illumination + sigma + self.eps)
        return torch.cat((red, blue, illumination, contrast), dim=1).to(rgb.dtype)


class PriorResidualBranch(nn.Module):
    """One cue encoder, one content-conditioned spatial gate, one residual scale."""
    def __init__(self, prior_channels, feature_channels, width=16):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(prior_channels, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1, groups=width), nn.SiLU(),
            nn.Conv2d(width, width, 1), nn.SiLU(),
        )

        self.content = nn.Conv2d(feature_channels, width, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(width * 2, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, 1, 1), nn.Sigmoid(),
        )

        self.project = nn.Conv2d(width, feature_channels, 1)
        self.scale = nn.Parameter(torch.full((1, feature_channels, 1, 1), 1e-3))

    def forward(self, feature, cue, reliability=None):
        encoded = self.encode(cue)
        gate = self.gate(torch.cat((self.content(feature), encoded), dim=1))
        if reliability is not None:
            gate = gate * reliability
        residual = self.scale * gate * self.project(encoded)
        return residual, gate


class ComplementaryPriorFusion(nn.Module):
    def __init__(self, feature_channels, prior_types=tuple(PRIOR_CHANNELS),
                 width=16, dark_floor=0.05):
        super().__init__()
        if width < 1 or dark_floor <= 0:
            raise ValueError("width and dark_floor must be positive")
        self.prior_types = normalize_prior_types(prior_types)
        self.dark_floor = dark_floor
        self.branches = nn.ModuleDict({
            name: PriorResidualBranch(PRIOR_CHANNELS[name][1] - PRIOR_CHANNELS[name][0],
                                      feature_channels, width)
            for name in self.prior_types
        })

    def forward(self, feature, prior_maps):
        # Each decoder scale encodes raw cues independently; no upsampled H/4 feature
        # is reused as the H/2 prior. Area pooling reduces fine-scale aliasing.
        cues = F.adaptive_avg_pool2d(prior_maps, feature.shape[-2:])
        illumination = cues[:, 2:3]
        # A bounded dark-region attenuation, not a calibrated uncertainty estimate.
        color_reliability = illumination / (illumination + self.dark_floor)
        residual = torch.zeros_like(feature)
        gates = {}
        for name, branch in self.branches.items():
            start, end = PRIOR_CHANNELS[name]
            update, gate = branch(feature, cues[:, start:end],
                                  color_reliability if name == "color" else None)
            residual = residual + update
            gates[name] = gate
        return feature + residual, gates
