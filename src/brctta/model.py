"""Source model wrapper + the residual adapter that test-time adaptation updates.

Design constraint (spec §4): the state ``theta`` must be *small enough to roll
back exactly* and must support the three actions keep / accept / reset.  We
therefore keep the MambaCue backbone frozen and put all plasticity into a small
residual adapter::

    y = clamp( backbone(x) + adapter(x, backbone(x)), 0, 1 )

The adapter's last layer is zero-initialised, so ``theta_0`` (adapter = 0) is
exactly the frozen source model, and ``reset`` is the identity operation on
theta rather than a snapshot reload.  Accepting an update stores the adapter
weights; rolling back restores them byte-for-byte, which is what the replay
check verifies.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import backbones as BB


# --------------------------------------------------------------------------- #
# adapter
# --------------------------------------------------------------------------- #
class ResidualAdapter(nn.Module):
    """Small, zero-initialised residual corrector.  ~13k parameters.

    Two mechanisms, both cheap and both reversible:
      * a 2-layer 3x3 conv stack over ``[x, y_source]`` -> colour/detail residual
      * a global affine (gamma, beta) predicted from pooled``[x, y_source]``
        which models the global colour cast typical of underwater scenes
    """

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(6, hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden, 3, 3, padding=1)
        self.affine = nn.Linear(6, 6)
        self.act = nn.GELU()
        for m in (self.conv3, self.affine):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(torch.cat([x, y], dim=1)))
        h = self.act(self.conv2(h))
        residual = self.conv3(h)
        stats = torch.cat([x.mean(dim=(2, 3)), y.mean(dim=(2, 3))], dim=1)
        gamma, beta = self.affine(stats).chunk(2, dim=1)
        affine = y * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return residual + (affine - y)


class AdaptedModel(nn.Module):
    """Frozen MambaCue + trainable adapter, with explicit state control."""

    def __init__(self, backbone: nn.Module, hidden: int = 32, device: str = "cuda"):
        super().__init__()
        self.backbone = backbone
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.adapter = ResidualAdapter(hidden).to(device)
        self.device = device

    # -- state --------------------------------------------------------------
    def snapshot(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.adapter.state_dict().items()}

    def restore(self, state: Dict[str, torch.Tensor]) -> None:
        self.adapter.load_state_dict({k: v.to(self.device) for k, v in state.items()})

    def adapter_params(self):
        return [p for p in self.adapter.parameters() if p.requires_grad]

    def update_magnitude(self, reference: Dict[str, torch.Tensor]) -> float:
        """L2 distance between current adapter weights and a reference snapshot."""
        total = 0.0
        cur = self.adapter.state_dict()
        for k, v in cur.items():
            total += float(((v.detach() - reference[k]) ** 2).sum())
        return total ** 0.5

    # -- inference ----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gradient-capable by design: the candidate update differentiates through
        this path.  Callers that only need inference wrap it in ``torch.no_grad``
        themselves (see ``AdaptedModel.predict``)."""
        y = self.backbone(x)
        out = y + self.adapter(x, y)
        return out.clamp(0.0, 1.0)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def build_backbone(ckpt_path: str, device: str = "cuda", kind: str = "mambacue") -> nn.Module:
    """Frozen backbone by name (see ``brctta.backbones``).

    ``kind="aqua"`` lets the same risk-controlled adapter be measured on an
    independently trained public model, so the comparison set is not restricted
    to a single (and comparatively small) backbone.
    """
    return BB.build_backbone(kind, ckpt_path, device=device)


def load_adapter(model: AdaptedModel, adapter_ckpt: str) -> AdaptedModel:
    """Install a pre-trained adapter as ``theta_0``.

    The adapter in ``adapter_ckpt`` was trained on SOURCE pairs only
    (scripts/01b_adapter_pretrain.py), so the resulting theta_0 is still a
    source-only model under spec §5.1 -- it simply carries a learned residual
    instead of starting at exactly zero.  Every method that runs on top of it
    starts from the same theta_0, so the comparison is unaffected; what changes
    is that "do nothing" is now a strong reference rather than a weak one.
    """
    payload = torch.load(adapter_ckpt, map_location="cpu")
    state = payload.get("adapter", payload)
    model.adapter.load_state_dict(state)
    meta = {k: v for k, v in payload.items() if k != "adapter"}
    model.adapter_meta = meta
    return model


def build_adapted(
    ckpt_path: str,
    device: str = "cuda",
    hidden: int = 32,
    backbone: str = "mambacue",
    adapter_ckpt: Optional[str] = None,
) -> AdaptedModel:
    model = AdaptedModel(build_backbone(ckpt_path, device, kind=backbone),
                         hidden=hidden, device=device)
    if adapter_ckpt:
        load_adapter(model, adapter_ckpt)
    else:
        model.adapter_meta = {}
    return model
