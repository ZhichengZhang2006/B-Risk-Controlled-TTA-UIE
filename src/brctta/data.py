"""Deterministic data access for the *locked* main protocol.

Protocol facts fixed here (spec §5.1):
  * Evaluation resolution is 256x256 for every image: the WHOLE image is resized
    (bicubic for inputs, bicubic for references), never cropped and never padded.
  * Both members of a pair are resized independently to 256x256, so a
    super-resolution dataset (UFO-120: 320x240 input vs 640x480 GT) is reduced to
    a same-size restoration pair. This is deliberate; it is recorded in the data
    manifest so nobody can later claim SR gains.
  * Orders are fixed by a seeded permutation of the sorted file stem list, so all
    methods and all baselines consume exactly the same stream.
  * No image is ever dropped: the main table scores every target image.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

EVAL_SIZE = 256
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp",
           ".PNG", ".JPG", ".JPEG", ".BMP", ".TIF", ".TIFF", ".WEBP")


# --------------------------------------------------------------------------- #
# deterministic stream definition
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StreamSpec:
    """One locked evaluation/adaptation stream."""

    name: str                 # e.g. "LSUI2UIEB"
    source_dataset: str       # dataset used for source supervision
    target_split: str         # "UIEB/test"
    seed: int = 0
    order: str = "shuffle"    # "shuffle" | "natural"
    eval_size: int = EVAL_SIZE

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


def list_stems(split_dir: str) -> List[str]:
    """Paired stems present in both ``input/`` and ``GT/``, sorted."""
    inp = os.path.join(split_dir, "input")
    gt = os.path.join(split_dir, "GT")
    extract = lambda d: {os.path.splitext(f)[0] for f in os.listdir(d) if f.lower().endswith(IMG_EXT)}
    stems = sorted(extract(inp) & extract(gt))
    if not stems:
        raise FileNotFoundError(f"no paired images under {split_dir}")
    return stems


def resolve_path(split_dir: str, stem: str, sub: str) -> str:
    """Locate ``stem`` in ``split_dir/sub`` tolerating mixed-case extensions.

    Some datasets (EUVP-d) mix ``.jpg``/``.JPEG``; the stem is what identifies a
    pair, so the lookup is case-insensitive on the suffix and exact on the stem.
    """
    folder = os.path.join(split_dir, sub)
    for ext in IMG_EXT:
        candidate = os.path.join(folder, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    low = stem.lower()
    for fname in os.listdir(folder):
        root, ext = os.path.splitext(fname)
        if root == stem or (root.lower() == low and ext.lower() in [e.lower() for e in IMG_EXT]):
            return os.path.join(folder, fname)
    raise FileNotFoundError(f"{stem} not found in {folder}")


def _index_folder(folder: str) -> Dict[str, str]:
    """stem -> full path, built once. Keys are both the exact and lower-case stem."""
    idx: Dict[str, str] = {}
    for fname in os.listdir(folder):
        root, ext = os.path.splitext(fname)
        if ext.lower() not in IMG_EXT:
            continue
        full = os.path.join(folder, fname)
        idx.setdefault(root, full)
        idx.setdefault(root.lower(), full)
    return idx


def stream_order(stems: Sequence[str], spec: StreamSpec) -> List[str]:
    if spec.order == "natural":
        return list(stems)
    rng = np.random.RandomState(spec.seed)
    idx = rng.permutation(len(stems))
    return [stems[i] for i in idx]


def file_sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# resampling: one implementation for everyone
# --------------------------------------------------------------------------- #
def load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_eval(img: Image.Image, size: int = EVAL_SIZE) -> Image.Image:
    """Whole-image resize, no crop, no pad, bicubic, antialiased corners.

    ``Image.resize`` with BICUBIC on a downscale does *not* prefilter unless
    ``reducing_gap`` is used, so we set it to obtain a consistent, reproducible
    antialiasing behaviour across Pillow versions.
    """
    if img.size == (size, size):
        return img
    return img.resize((size, size), Image.BICUBIC, reducing_gap=3.0)


def to_tensor(img: Image.Image) -> torch.Tensor:
    """uint8 HWC -> float32 CHW in [0,1].

    Done in numpy on purpose: torch's default intra-op pool is sized to the CPU
    count (48 here), and for a 196 kB tensor the synchronisation costs ~50x the
    arithmetic (measured 15.2 ms vs 0.27 ms, scripts/_micro_probe.py).
    """
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))



class PairedStream(Dataset):
    """Ordered paired stream. Yields tensors of shape [3, S, S] in [0, 1]."""

    def __init__(self, split_dir: str, spec: StreamSpec, cache: bool = True):
        self.split_dir = split_dir
        self.spec = spec
        self.stems = stream_order(list_stems(split_dir), spec)
        self.cache = cache
        self._memo: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        # resolve_path() re-lists the directory on every call (it exists to
        # tolerate mixed-case extensions).  That was measured at ~110 ms per
        # image on NFS, so index the two folders once and reuse the mapping.
        self._index = {"input": _index_folder(os.path.join(split_dir, "input")),
                       "GT": _index_folder(os.path.join(split_dir, "GT"))}

    def __len__(self) -> int:
        return len(self.stems)

    def _path(self, stem: str, sub: str) -> str:
        hit = self._index[sub].get(stem)
        if hit is not None:
            return hit
        return resolve_path(self.split_dir, stem, sub)

    def _load(self, stem: str):
        if self.cache and stem in self._memo:
            return self._memo[stem]
        x = to_tensor(resize_eval(load_rgb(self._path(stem, "input")), self.spec.eval_size))
        y = to_tensor(resize_eval(load_rgb(self._path(stem, "GT")), self.spec.eval_size))
        if self.cache:
            self._memo[stem] = (x, y)
        return x, y

    def __getitem__(self, index: int):
        stem = self.stems[index]
        x, y = self._load(stem)
        return {"input": x, "gt": y, "stem": stem, "index": index}

    def take(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        x, y = self._load(self.stems[index])
        return x, y, self.stems[index]


def collate(items):
    return {
        "input": torch.stack([it["input"] for it in items]),
        "gt": torch.stack([it["gt"] for it in items]),
        "stem": [it["stem"] for it in items],
        "index": torch.tensor([it["index"] for it in items], dtype=torch.long),
    }


# --------------------------------------------------------------------------- #
# manifests
# --------------------------------------------------------------------------- #
def build_manifest(split_dir: str, spec: StreamSpec, verify_hash: bool = False) -> dict:
    """Data provenance record: counts, order, sizes, optional content hashes."""
    stems = list_stems(split_dir)
    ordered = stream_order(stems, spec)
    sizes = {}
    tries = {0: np.asarray(load_rgb(resolve_path(split_dir, ordered[0], "input")).size),}
    for sub in ("input", "GT"):
        tries[sub] = np.asarray(load_rgb(resolve_path(split_dir, ordered[0], sub)).size)
    entry = {
        "stream": asdict(spec),
        "stream_digest": spec.digest(),
        "split_dir": split_dir,
        "num_pairs": len(stems),
        "order_sha256": hashlib.sha256("\n".join(ordered).encode()).hexdigest(),
        "first_ten": ordered[:10],
        "probe_sizes_input": tries["input"].tolist(),
        "probe_sizes_gt": tries["GT"].tolist(),
        "resize_to": spec.eval_size,
        "resampling": "PIL BICUBIC whole-image, reducing_gap=3.0, no crop, no pad",
    }
    if verify_hash:
        entry["input_sha256"] = {s: file_sha256(resolve_path(split_dir, s, "input")) for s in ordered}
        entry["gt_sha256"] = {s: file_sha256(resolve_path(split_dir, s, "GT")) for s in ordered}
    return entry
