"""Common runtime helpers: config loading, paths, bookkeeping."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))


# The pod has 96 CPUs but 16 GB RAM shared with other tenants.  torch sizes its
# intra-op pool from the CPU count, so a single 196 kB tensor conversion spends
# 15 ms in synchronisation against 0.3 ms of work (measured, scripts/_micro_probe.py).
# A small cap keeps host work predictable; the GPU is the resource we care about.
TORCH_THREADS = int(os.environ.get("BRCTTA_TORCH_THREADS", "4"))


def init_torch() -> None:
    try:
        import torch
        torch.set_num_threads(TORCH_THREADS)
        torch.set_num_interop_threads(1)
    except Exception:  # noqa: BLE001 -- never let a cosmetic setting kill a run
        pass


def load_config(path: str | None = None) -> Dict:
    path = path or os.path.join(ROOT, "configs", "main.yaml")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = ROOT
    init_torch()
    return cfg


def split_dir(cfg: Dict, ds: str, split: str) -> str:
    rel = cfg["splits"][ds][split]
    return os.path.join(cfg["paths"]["data_root"], rel)


def ckpt_path(cfg: Dict, rel: str) -> str:
    """Resolve a checkpoint reference.

    Accepts either ``LSUI/best_model.pth`` or a full path.  Checkpoints live in
    timestamped sub-directories upstream, so a glob fallback picks the newest.
    """
    if os.path.isabs(rel) and os.path.exists(rel):
        return rel
    full = os.path.join(cfg["paths"]["weights_root"], rel)
    if os.path.exists(full):
        return full
    import glob

    hits = sorted(glob.glob(os.path.join(cfg["paths"]["weights_root"], rel.replace("best_model.pth", "*", 1), "best_model.pth")))
    if hits:
        return hits[-1]
    hits = sorted(glob.glob(os.path.join(cfg["paths"]["weights_root"], rel, "*/best_model.pth")))
    if hits:
        return hits[-1]
    raise FileNotFoundError(f"checkpoint not found for ref {rel!r} under {cfg['paths']['weights_root']}")


AQUA_ROOT = "/home/user/AquaClarity-main"
_AQUA_DIR = {"UIEB": "UIEB", "LSUI": "LSUI", "UFO": "UFO", "EUVPd": "EUVP-d"}


def aqua_ckpt(ds: str) -> str:
    """Published AquaClarity checkpoint trained on ``ds`` (an independent model)."""
    return os.path.join(AQUA_ROOT, "checkpoints", _AQUA_DIR.get(ds, ds), "best.pth")


def resolve_ckpt(cfg: Dict, stream_cfg: Dict, backbone: str = "mambacue") -> str:
    """Checkpoint for the frozen backbone of a stream.

    ``mambacue`` uses the weights this project trained/pinned; ``aqua`` switches
    to AquaClarity's published per-dataset checkpoint, which is what makes the
    §11 comparison set contain an independently trained strong model.
    """
    if backbone == "aqua":
        return aqua_ckpt(stream_cfg["source"])
    return ckpt_path(cfg, stream_cfg["source_ckpt"])


def adapter_ckpt_path(cfg: Dict, source: str, backbone: str = "mambacue") -> str | None:
    """theta_0's learned residual: the SOURCE-pretrained adapter, if it exists.

    Returns None when the pretrain step has not run, in which case theta_0 is
    the bare frozen backbone (still legal, just a weaker starting point).  The
    caller records which of the two it used, so a run can never be silently
    attributed to the wrong theta_0.
    """
    suf = "" if backbone == "mambacue" else f"_{backbone}"
    p = os.path.join(cfg["paths"]["out_root"], f"adapter_pretrain_{source}{suf}", "adapter.pth")
    return p if os.path.exists(p) else None


def degrader_path(cfg, source: str, backbone: str = "mambacue") -> str:
    return os.path.join(cfg["paths"]["out_root"], f"degrader_{source}", "degrader.pth")


def load_degrader(cfg, source: str, device: str, backbone: str = "mambacue"):
    """Load the SOURCE-fitted degradation model D: clean -> degraded.

    Returns ``(net, path)`` or ``(None, path)`` when it has not been trained yet.
    Callers must tolerate ``None`` (the evidence keys then stay absent and the
    controller sees NaN -> 0.0), so the pipeline still runs end-to-end before the
    degrader exists; but a run without it is *not* the final method, and
    ``05_online_eval`` records which case it was in ``online_results.json``.
    """
    path = degrader_path(cfg, source, backbone)
    if not os.path.exists(path):
        return None, path
    import torch
    from brctta.degrade import Degrader
    payload = torch.load(path, map_location="cpu")
    net = Degrader(width=int(payload.get("width", 32)))
    net.load_state_dict(payload["degrader"])
    net.to(device).eval()
    for p_ in net.parameters():
        p_.requires_grad_(False)
    return net, path


def theta0_label(cfg: Dict, source: str, backbone: str = "mambacue") -> str:
    return ("source_adapter" if adapter_ckpt_path(cfg, source, backbone) else "frozen_backbone")


def backbone_suffix(backbone: str) -> str:
    """Directory suffix so runs from different backbones never overwrite."""
    return "" if backbone == "mambacue" else f"_{backbone}"


def cache_root(cfg: Dict) -> str:
    return cfg["paths"].get("cache_root") or os.path.join(cfg["paths"]["out_root"], "_cache")


def open_stream(cfg: Dict, ds: str, split: str, spec, use_cache: bool = True):
    """Cached (fast) stream for one split; falls back to the plain loader.

    The measurement that motivates this: decoding+resizing one LSUI pair costs
    ~300 ms cold (page-cache warm: 24 ms), so a 8-image batch spends ~4.8 s on
    host work against ~0.21 s of GPU.  Caching is a memoisation of the identical
    protocol tensor (verified bitwise by cache.verify_against_loader), not a
    change to it.
    """
    from brctta.cache import open_stream as _open
    root = split_dir(cfg, ds, split)
    return _open(root, spec, cache_root(cfg) if use_cache else None)


def run_dir(cfg: Dict, stream: str, tag: str = "") -> str:
    d = os.path.join(cfg["paths"]["out_root"], stream + (f"_{tag}" if tag else ""))
    os.makedirs(d, exist_ok=True)
    return d


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.seconds = time.time() - self.t0
        return False
