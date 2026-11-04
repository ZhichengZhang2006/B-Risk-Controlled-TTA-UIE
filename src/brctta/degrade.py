"""A learned underwater *degradation* model, fitted on SOURCE pairs only.

Why this exists
---------------
Every no-reference objective tried so far (`hf`, `lab`, `rgb`, `gamma`, `dcp`,
`saturation`) either loses PSNR outright or wins only by a hair.  They all share
one weakness: they constrain the *statistics* of the output, not its
*relationship to the input*.  A statistic can be satisfied by an image that no
longer depicts the same scene.

The degradation model supplies exactly the missing relation.  Fitted on source
pairs ``(x_s, g_s)`` as ``D: g_s -> x_s``, it answers "if this output were a
clean image of that scene, would re-degrading it reproduce what the camera
actually saw?".  The residual

    r(y) = || D(y) - x ||_1

is a per-image, label-free quantity that is zero only when ``y`` is a clean
image consistent with the observed degradation of ``x``.  It is the
"re-degradation residual" evidence listed in spec §7.2.

`D` is deliberately small and bounded: a per-pixel attenuation/backscatter head
``D(g) = g * a(g) + b(g)`` where ``a``, ``b`` are predicted from a 3-level
convolutional encoder.  This is the revised underwater image formation model
(``I = J * t + B * (1 - t)``) with ``t`` and ``B`` predicted per pixel, which
keeps the map monotone in intensity and prevents the degenerate solution of
learning an arbitrary invertible warp.  It also fits in ~1 minute on one GPU,
so the whole risk pipeline stays inside the §12 budget.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(4, cout),
        nn.SiLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(4, cout),
        nn.SiLU(inplace=True),
    )


class Degrader(nn.Module):
    """Predicts per-pixel transmission ``t`` and backscatter ``B``.

    ``D(g) = g * t + B * (1 - t)`` with ``t = sigmoid(raw_t)`` in (0,1) and
    ``B`` a per-pixel colour, both predicted from ``g``.  Initialised near the
    identity-degradation regime (``t≈0.9``, ``B≈0``) so early training cannot
    collapse to a constant map.
    """

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.enc1 = _block(3, width)
        self.enc2 = _block(width, width * 2)
        self.enc3 = _block(width * 2, width * 4)
        self.dec2 = _block(width * 4 + width * 2, width * 2)
        self.dec1 = _block(width * 2 + width, width)
        self.head = nn.Conv2d(width, 6, 3, padding=1)  # 3 for t, 3 for B
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # bias so that sigmoid(raw_t)=0.9 for t and B=0.5 for the colour channel
        with torch.no_grad():
            self.head.bias[:3] = 2.197  # logit(0.9)
            self.head.bias[3:] = 0.0

    def forward(self, g: torch.Tensor):
        e1 = self.enc1(g)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([F.interpolate(e3, size=e2.shape[-2:], mode="nearest"), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="nearest"), e1], 1))
        out = self.head(d1)
        t = torch.sigmoid(out[:, :3])
        b = torch.sigmoid(out[:, 3:])
        return g * t + b * (1.0 - t)

    def decompose(self, g: torch.Tensor):
        """Returns (t, B) so a caller can inspect the fitted water model."""
        e1 = self.enc1(g)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([F.interpolate(e3, size=e2.shape[-2:], mode="nearest"), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="nearest"), e1], 1))
        out = self.head(d1)
        return torch.sigmoid(out[:, :3]), torch.sigmoid(out[:, 3:])


def redegradation_loss(degrader: Degrader, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """``||D(y) - x||_1`` -- how badly the candidate output fails to explain the input.

    Note this is ONE-SIDED: it never rewards making ``y`` darker or brighter for
    its own sake, only for reproducing the observed degradation.  That is the
    property the statistics priors lacked.
    """
    return (degrader(y) - x).abs().mean()
