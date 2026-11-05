"""Self-supervised objectives used for the candidate update (no labels).

Underwater enhancement has no target labels at deployment, but the source domain
provides two legitimate offline artefacts that a target-free update may use:

  1. a *physical* prior: x = J*t + A*(1-t) with a smooth transmission map, so the
     difference (x - J) should be dominated by low frequencies;
  2. a *statistical* prior: the source clean images occupy a known colour
     distribution (per-channel mean/std, Lab a/b), computed once from source GT.

Any label-free objective built from these is a proxy.  Proxies sometimes help and
sometimes hurt -- which is exactly why the risk controller exists (spec §4/§7.2).
"""

from __future__ import annotations

import json
from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _highpass(x: torch.Tensor, k: int = 5) -> torch.Tensor:
    pad = k // 2
    blur = F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), k, stride=1)
    return x - blur


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    return (x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (x[..., :, 1:] - x[..., :, :-1]).abs().mean()


def hf_residual_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The input/restoration difference should be dominated by low frequency."""
    return _highpass(x - y).abs().mean() / (_highpass(x).abs().mean() + 1e-6)


def rgb_prior_loss(y: torch.Tensor, prior: Dict) -> torch.Tensor:
    """Match per-image channel statistics to the source clean-image prior."""
    mu = torch.tensor(prior["rgb_mean"], device=y.device, dtype=y.dtype)[None, :, None, None]
    sd = torch.tensor(prior["rgb_std"], device=y.device, dtype=y.dtype)[None, :, None, None]
    img_mu = y.mean(dim=(2, 3), keepdim=True)
    img_sd = y.std(dim=(2, 3), keepdim=True) + 1e-6
    return ((img_mu - mu) ** 2).mean() * 2.0 + ((img_sd - sd) ** 2).mean() * 0.5


def lab_ab_prior_loss(y: torch.Tensor, prior: Dict) -> torch.Tensor:
    """Pull the CIE-Lab a/b means toward the source prior (colour-cast removal).

    Implemented with a fixed linear sRGB->XYZ->Lab approximation so it stays
    differentiable and cheap; the exact Lab conversion is used only for metrics.
    """
    if "lab_ab_mean" not in prior:
        return torch.zeros((), device=y.device, dtype=y.dtype)

    r, g, b = y[:, 0], y[:, 1], y[:, 2]
    X = 0.4124 * r + 0.3576 * g + 0.1805 * b
    Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    Z = 0.0193 * r + 0.1192 * g + 0.9505 * b
    xr, yr, zr = X / 0.9505, Y, Z / 1.089
    f = lambda t: torch.where(t > 0.008856, t.pow(1 / 3), 7.787 * t + 16 / 116)
    fx, fy, fz = f(xr), f(yr), f(zr)
    a = 500.0 * (fx - fy)
    bb = 200.0 * (fy - fz)

    tgt = torch.tensor(prior["lab_ab_mean"], device=y.device, dtype=y.dtype)
    ma, mb = a.mean(), bb.mean()
    return ((ma - tgt[0]) ** 2 + (mb - tgt[1]) ** 2) / 100.0


def consistency_loss(model, x: torch.Tensor, aug: str = "photometric",
                     generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Augmentation consistency: f(x) should commute with a photometric jitter.

    ``generator`` makes the jitter reproducible.  Without it the objective draws
    ``torch.empty(...).uniform_()`` from the global RNG, so two runs of the same
    stream with the same actions produce different candidates -- which is exactly
    what made the §13 replay check fail (max |dPSNR| 1.7e-3 dB on UIEB2LSUI,
    2.2e-4 dB on LSUI2UFO).  Sampling on CPU from an explicit generator, then
    moving the two scalars to the device, keeps the sample identical across
    processes and devices.
    """
    if aug == "photometric":
        if generator is not None:
            b = torch.empty(x.shape[0], 1, 1, 1).uniform_(-0.10, 0.10, generator=generator).to(x.device)
            c = torch.empty(x.shape[0], 1, 1, 1).uniform_(0.85, 1.15, generator=generator).to(x.device)
        else:
            b = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(-0.10, 0.10)
            c = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(0.85, 1.15)
        xp = (x * c + b).clamp(0.0, 1.0)
        y = model(x)
        yp = model(xp)
        # compare in the jittered domain so the residual measures invariance
        return ((y * c + b).clamp(0.0, 1.0) - yp).abs().mean()
    y = model(x)
    yp = model(x.flip(-1))
    return 0.5 * (y - yp.flip(-1)).abs().mean() + 0.5 * (yp - y.flip(-1)).abs().mean()


def self_supervised_loss(
    model,
    x: torch.Tensor,
    prior: Optional[Dict] = None,
    weights: Optional[Dict[str, float]] = None,
    with_consistency: bool = True,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Composite no-label objective driving the candidate update."""
    w = {"hf": 1.0, "rgb": 0.5, "lab": 0.5, "tv": 0.02, "cons": 0.30}
    if weights:
        w.update(weights)

    y = model(x)
    loss = w["hf"] * hf_residual_loss(x, y)
    if prior:
        loss = loss + w["rgb"] * rgb_prior_loss(y, prior)
        loss = loss + w["lab"] * lab_ab_prior_loss(y, prior)
    loss = loss + w["tv"] * tv_loss(y)
    if with_consistency and w["cons"] > 0:
        loss = loss + w["cons"] * consistency_loss(model, x, generator=generator)
    return loss


def load_prior(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
