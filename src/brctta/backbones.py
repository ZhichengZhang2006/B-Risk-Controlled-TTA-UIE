"""Pluggable frozen backbones.

The risk-controlled adapter is deliberately backbone-agnostic: it only needs a
frozen map ``x -> y`` in [0, 1].  Two backbones are registered so the same
mechanism can be measured on more than one model (spec §11 forbids letting a
single weak backbone stand in for the whole comparison set):

* ``mambacue`` -- our own MambaCue, vendor-copied into ``brctta.mambacue``.
* ``aqua``     -- AquaClarity (NILUT + Uformer), an independent public model with
  published checkpoints for every dataset we use.  Loaded lazily from
  ``/home/user/AquaClarity-main`` so this repo keeps working without it.

Both expose ``forward(x) -> y`` and are put in eval() with all parameters frozen.
"""

from __future__ import annotations

import os
import sys
from typing import Dict

import torch
import torch.nn as nn

from .mambacue import myModel

AQUA_ROOT = "/home/user/AquaClarity-main"
BACKBONES = ("mambacue", "aqua")


class ClampedWrapper(nn.Module):
    """Force a backbone's output into [0, 1] and keep the module introspectable."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).clamp(0.0, 1.0)


def _load_state(path: str) -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    out = {}
    for k, v in state.items():
        name = k
        for prefix in ("module.", "_orig_mod."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        out[name] = v
    return out


def build_mambacue(ckpt_path: str, device: str = "cuda") -> nn.Module:
    model = myModel(
        in_channels=3, hidden_channels=32, out_channels=3,
        prior_types=("color", "illumination", "contrast"),
        scan_mode="four", use_checkpoint=False, output_heads="both",
    )
    model.load_state_dict(_load_state(ckpt_path), strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build_aqua(ckpt_path: str, device: str = "cuda") -> nn.Module:
    if AQUA_ROOT not in sys.path:
        sys.path.insert(0, AQUA_ROOT)
    from models.Aqua import Model  # noqa: E402

    model = Model()
    missing, unexpected = model.load_state_dict(_load_state(ckpt_path), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"aqua state_dict mismatch: missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}"
        )
    model = ClampedWrapper(model).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build_backbone(kind: str, ckpt_path: str, device: str = "cuda") -> nn.Module:
    if kind == "mambacue":
        return build_mambacue(ckpt_path, device)
    if kind == "aqua":
        return build_aqua(ckpt_path, device)
    raise ValueError(f"unknown backbone {kind!r}; expected one of {BACKBONES}")


def parameter_counts(net: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in net.parameters())
    train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(train)}
