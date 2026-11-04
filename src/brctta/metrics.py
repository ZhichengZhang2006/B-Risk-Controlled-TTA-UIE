"""Per-image quality metrics, computed identically for every method.

The main table reports PSNR and SSIM at 256x256 (spec §5.1).  MSE is also kept
because §11 forbids substituting an MSE-based criterion for the PSNR/SSIM one --
we report all three, and the acceptance gate is evaluated on PSNR/SSIM only.

All functions take float tensors/arrays in [0, 1] with shape [3, H, W].
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch


def _to_np(x) -> np.ndarray:
    """Normalise any accepted input shape to [H, W, 3] float64 in [0, 1].

    Accepts [3,H,W], [1,3,H,W] and [H,W,3].  Dropping the leading batch axis is
    not cosmetic: ``ssim`` indexes ``im[..., c]`` and passes ``sigma=(1.5,1.5)``
    to ``gaussian_filter``, which requires the array rank to equal the sigma
    length.  A [1,3,H,W] input previously reached that call as a 4-D array and
    died with ``RuntimeError: sequence argument must have length equal to input
    rank`` -- while ``psnr`` silently kept working, because ``np.mean`` does not
    care about the extra axis.  That asymmetry is exactly the kind of thing that
    makes one metric wrong and another right without anyone noticing.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    x = np.asarray(x, dtype=np.float64)
    while x.ndim > 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == 3 and x.shape[0] == 3:
        x = x.transpose(1, 2, 0)
    if x.ndim != 3:
        raise ValueError(f"expected [3,H,W] or [1,3,H,W], got shape {x.shape}")
    return np.clip(x, 0.0, 1.0)


def psnr(pred, gt, data_range: float = 1.0) -> float:
    p, g = _to_np(pred), _to_np(gt)
    mse = float(np.mean((p - g) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(data_range ** 2 / mse))


def mse(pred, gt) -> float:
    p, g = _to_np(pred), _to_np(gt)
    return float(np.mean((p - g) ** 2))


def ssim(pred, gt, data_range: float = 1.0, win_size: int = 11) -> float:
    """Gaussian-window SSIM matching the common UIE reporting convention.

    Implemented with scipy only (no skimage dependency at eval time) so the
    number cannot drift with library versions.  Uses a unit-sum Gaussian of
    sigma=1.5, cropped to ``win_size``, and excludes border pixels.
    """
    from scipy.ndimage import gaussian_filter

    p, g = _to_np(pred), _to_np(gt)
    half = win_size // 2
    # The kernel radius must come out at exactly ``half`` so that cropping the
    # border by ``half`` removes precisely the region with no valid support.
    # scipy's default truncate=4.0 would build a 13x13 kernel for sigma=1.5
    # (radius int(4.0*1.5+0.5)=6) while we crop 5 -- measured 1.65e-4 SSIM
    # disagreement with skimage's canonical Wang implementation.  truncate is
    # therefore pinned to (win-1)/(2*sigma) = 3.333, which gives radius 5.
    trunc = (win_size - 1) / (2 * 1.5)
    filt = lambda im: np.stack(
        [gaussian_filter(im[..., c], sigma=(1.5, 1.5), mode="nearest", truncate=trunc)
         for c in range(im.shape[-1])],
        axis=-1,
    )
    mu_p, mu_g = filt(p), filt(g)
    sigma_pp = filt(p * p) - mu_p ** 2
    sigma_gg = filt(g * g) - mu_g ** 2
    sigma_pg = filt(p * g) - mu_p * mu_g
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    s = ((2 * mu_p * mu_g + c1) * (2 * sigma_pg + c2)) / (
        (mu_p ** 2 + mu_g ** 2 + c1) * (sigma_pp + sigma_gg + c2)
    )
    if half > 0:
        s = s[half:-half, half:-half, :]
    return float(s.mean())


def per_image(pred, gt) -> Dict[str, float]:
    return {"psnr": psnr(pred, gt), "ssim": ssim(pred, gt), "mse": mse(pred, gt)}


def mean_of(rows: Sequence[Dict[str, float]], keys: Iterable[str] = ("psnr", "ssim", "mse")):
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def median_of(rows: Sequence[Dict[str, float]], keys: Iterable[str] = ("psnr", "ssim")):
    return {k: float(np.median([r[k] for r in rows])) for k in keys}


def worst_fraction_mean(rows: Sequence[Dict[str, float]], key: str, frac: float = 0.10) -> float:
    vals = np.sort(np.asarray([r[key] for r in rows], dtype=np.float64))
    k = max(1, int(round(len(vals) * frac)))
    return float(vals[:k].mean())


# --------------------------------------------------------------------------- #
# no-reference underwater metrics (reported as secondary evidence only)
# --------------------------------------------------------------------------- #
def uiqm(pred) -> float:
    """Underwater Image Quality Measure (Panetta et al.), a common UIE side metric."""
    from scipy.ndimage import uniform_filter

    im = _to_np(pred)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b

    def _sharpness(ch):
        gx = np.diff(ch, axis=1)
        gy = np.diff(ch, axis=0)
        return float(np.mean(np.abs(gx)) + np.mean(np.abs(gy)))

    def _contrast(ch):
        return float(np.std(ch))

    def _colorfulness():
        rg = r - g
        yb = 0.5 * (r + g) - b
        return float(np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
                     + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2))

    eme = float(np.mean([_sharpness(c) for c in (r, g, b)]))
    con = float(np.mean([_contrast(c) for c in (r, g, b)]))
    col = _colorfulness()
    # Panetta's linear combination (UIQM = 3.32*UICM + 0.73*UISM + 0.53*UIQM_color)
    return float(3.32 * col + 0.73 * eme + 0.53 * con)


def uciqe(pred) -> float:
    """UCIQE (Yang & Sowmya): 0.4680*sigma_C + 0.2745*con_l + 0.2576*mu_s, CIE-Lab."""
    from skimage import color as skcolor

    im = _to_np(pred)
    lab = skcolor.rgb2lab(im)
    L = lab[..., 0] / 100.0
    a = lab[..., 1]
    b = lab[..., 2]
    chroma = np.sqrt(a ** 2 + b ** 2)
    sigma_c = float(np.std(chroma))
    l_mean = float(np.mean(L))
    l_p99 = float(np.percentile(L, 99))
    l_p1 = float(np.percentile(L, 1))
    con_l = (l_p99 - l_p1) / max(l_mean, 1e-6)
    s = chroma / (chroma.max() + 1e-8)
    mu_s = float(np.mean(s))
    return float(0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s)


class MetricAccumulator:
    """Accumulates per-image metric rows; returns means and medians."""

    def __init__(self, no_ref: bool = False):
        self.no_ref = no_ref
        self.rows: List[Dict[str, float]] = []

    def add(self, pred, gt=None):
        if self.no_ref or gt is None:
            self.rows.append({"uiqm": uiqm(pred), "uciqe": uciqe(pred)})
        else:
            row = per_image(pred, gt)
            row.update({"uiqm": uiqm(pred), "uciqe": uciqe(pred)})
            self.rows.append(row)

    def summary(self) -> Dict[str, float]:
        if not self.rows:
            return {}
        out = mean_of(self.rows, keys=self.rows[0].keys())
        for k in ("psnr", "ssim"):
            if k in self.rows[0]:
                out[f"{k}_median"] = median_of(self.rows, keys=(k,))[k]
                out[f"{k}_worst10"] = worst_fraction_mean(self.rows, k, 0.10)
        out["n"] = len(self.rows)
        return out
