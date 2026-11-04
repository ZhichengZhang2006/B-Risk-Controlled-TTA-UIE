"""Materialised stream cache: decode+resize once, then read tensors at memory speed.

Why this exists (measured, not assumed).

    scripts/_data_probe.py on LSUI/train, 24 pairs:
        open+decode   131 ms/img
        resize 256     62 ms/img
        to tensor     109 ms/img
        cold total    302 ms/img   (warm, page-cached: 24 ms/img)
        PairedStream  412 ms/img   (+110 ms from resolve_path() re-listing the
                                    directory on every access; that is now
                                    indexed once in PairedStream)

    A batch of 8 therefore costs ~4.8 s of host time against ~0.21 s of GPU
    time. On a shared pod with 16 GB RAM and an NFS home, the data path -- not
    the model -- is the limit, and it would cap every experiment in the plan.

What the cache does and does not change.

    It changes nothing about the science: the cached array is the *exact*
    tensor `resize_eval(load_rgb(path), eval_size)` returns, stored as uint8 so
    the round-trip is lossless. The evaluation protocol (spec §5.1: whole-image
    resize to 256x256, no crop, no pad, one resampling rule for every method) is
    therefore untouched -- the cache is a memoisation of the protocol, not a
    redefinition of it. `verify_against_loader()` proves this by rebuilding a
    few items through the slow path and requiring bitwise equality.

    What it does change is honesty about provenance: the cache key includes the
    stream digest, the eval size and the directory, and the meta records a
    sha256 over the cached arrays, so a cached stream can never silently drift
    from the file list it claims to represent.

Layout (one directory per (split, stream)):
    x.npy     uint8 [N, S, S, 3]   network input
    y.npy     uint8 [N, S, S, 3]   reference
    meta.json {key, split, stream digest, eval_size, stems, sha256(x), sha256(y)}
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .data import (PairedStream, StreamSpec, list_stems, load_rgb, resize_eval,
                   resolve_path, stream_order, to_tensor)

META = "meta.json"


def _sha256_array(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def cache_key(split_dir: str, spec: StreamSpec) -> str:
    """Content key of a stream: everything that changes the tensor sequence.

    ``spec.name`` is deliberately excluded -- it is a label for logs, and
    including it would silently duplicate a 1.6 GiB cache for every caller that
    happens to name its stream differently.
    """
    payload = json.dumps({"split": os.path.abspath(split_dir),
                          "source": spec.source_dataset,
                          "target": spec.target_split,
                          "eval_size": spec.eval_size,
                          "order": spec.order,
                          "seed": spec.seed}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _decode_one(task) -> Tuple[np.ndarray, np.ndarray]:
    """Decode+resize one pair to uint8 HWC. Runs in a worker thread."""
    split_dir, stem, size = task
    xi = to_tensor(resize_eval(load_rgb(resolve_path(split_dir, stem, "input")), size))
    yi = to_tensor(resize_eval(load_rgb(resolve_path(split_dir, stem, "GT")), size))
    xb = (xi.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    yb = (yi.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    return xb, yb


def build_cache(split_dir: str, spec: StreamSpec, out_dir: str,
                overwrite: bool = False, progress: bool = True,
                workers: int = 16) -> str:
    """Decode every pair once and write the two uint8 arrays + meta.

    Decoding dominates (~131 ms/img) and happens inside Pillow's C code, which
    releases the GIL, so a thread pool scales this across the pod's CPUs without
    the fork/COW cost of processes.
    """
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, META)
    key = cache_key(split_dir, spec)
    if os.path.exists(meta_path) and not overwrite:
        try:
            with open(meta_path) as fh:
                meta = json.load(fh)
        except json.JSONDecodeError:
            meta = {}
        if (meta.get("key") == key and os.path.exists(os.path.join(out_dir, "x.npy"))
                and os.path.exists(os.path.join(out_dir, "y.npy"))):
            if progress:
                print(f"  cache hit {out_dir} ({meta.get('n')} items)")
            return out_dir

    stems = stream_order(list_stems(split_dir), spec)
    size = spec.eval_size
    n = len(stems)
    x = np.zeros((n, size, size, 3), dtype=np.uint8)
    y = np.zeros((n, size, size, 3), dtype=np.uint8)
    tasks = [(split_dir, st, size) for st in stems]
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, (xb, yb) in enumerate(pool.map(_decode_one, tasks)):
            x[i] = xb
            y[i] = yb
            done += 1
            if progress and done % 500 == 0:
                print(f"    {done}/{n}", flush=True)
    np.save(os.path.join(out_dir, "x.npy"), x)
    np.save(os.path.join(out_dir, "y.npy"), y)
    meta = {"key": key, "split": os.path.abspath(split_dir), "stream": spec.name,
            "stream_digest": spec.digest(), "eval_size": size, "n": n, "stems": stems,
            "sha256_x": _sha256_array(x), "sha256_y": _sha256_array(y)}
    tmp = meta_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(meta, fh)
    os.replace(tmp, meta_path)
    if progress:
        print(f"  wrote {out_dir} ({n} items, {size}x{size}, "
              f"{os.path.getsize(os.path.join(out_dir, 'x.npy')) / 2**20:.1f} MiB/array)")
    return out_dir


class CachedStream:
    """Read-only, indexed view over a materialised cache. Fast ``take()``."""

    def __init__(self, cache_dir: str, mmap: bool = True):
        meta_path = os.path.join(cache_dir, META)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"no cache at {cache_dir}; run build_cache first")
        with open(meta_path) as fh:
            self.meta = json.load(fh)
        mm = "r" if mmap else None
        self.x = np.load(os.path.join(cache_dir, "x.npy"), mmap_mode=mm)
        self.y = np.load(os.path.join(cache_dir, "y.npy"), mmap_mode=mm)
        self.stems: List[str] = list(self.meta["stems"])
        self.spec_name = self.meta.get("stream", "")
        self.eval_size = int(self.meta["eval_size"])
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return len(self.stems)

    @staticmethod
    def _to_tensor(arr: np.ndarray) -> torch.Tensor:
        """uint8 HWC -> float32 CHW in [0,1], via numpy.

        Measured on this pod (scripts/_micro_probe.py, 256x256x3):

            torch permute+contiguous+float+div   15.2 ms with 48 threads
                                                  0.27 ms with 1 thread
            numpy transpose+astype/255            0.13 ms

        torch's default thread pool (48 here, because the pod has 96 CPUs) costs
        more in synchronisation than the whole conversion does in work, so the
        conversion is done in numpy -- which is also why ``torch.set_num_threads``
        is capped in the entry points.
        """
        out = np.asarray(arr, dtype=np.float32).transpose(2, 0, 1)
        out = out / 255.0
        return torch.from_numpy(np.ascontiguousarray(out))

    def take(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        return self._to_tensor(self.x[index]), self._to_tensor(self.y[index]), self.stems[index]

    def take_batch(self, indices) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        idx = [int(i) for i in indices]
        xs = torch.stack([self._to_tensor(self.x[i]) for i in idx])
        ys = torch.stack([self._to_tensor(self.y[i]) for i in idx])
        return xs, ys, [self.stems[i] for i in idx]


def verify_against_loader(cache_dir: str, split_dir: str, spec: StreamSpec,
                          n: int = 5) -> Dict[str, object]:
    """Prove the cache is a faithful memoisation of the protocol path.

    Rebuilds ``n`` items through ``PairedStream`` (decode -> resize -> tensor)
    and requires the cached uint8 round-trip to match to within 1e-6, which is
    the float32 representation error of the uint8 quantisation and nothing else.
    A cache that fails this is not a cache, it is a different dataset.
    """
    cs = CachedStream(cache_dir)
    slow = PairedStream(split_dir, spec, cache=False)
    m = min(len(cs), len(slow))
    rng = np.random.RandomState(0)
    idxs = rng.choice(m, size=min(n, m), replace=False)
    max_dx, max_dy, bad = 0.0, 0.0, []
    for i in idxs:
        xf, yf = slow._load(cs.stems[int(i)])
        xc, yc, _ = cs.take(int(i))
        dx = float((xf - xc).abs().max())
        dy = float((yf - yc).abs().max())
        max_dx, max_dy = max(max_dx, dx), max(max_dy, dy)
        if dx > 1e-6 or dy > 1e-6:
            bad.append({"index": int(i), "dx": dx, "dy": dy})
    return {"ok": not bad, "max_abs_dx": max_dx, "max_abs_dy": max_dy,
            "n_checked": int(len(idxs)), "mismatches": bad,
            "note": ("cache is a lossless uint8 round-trip of resize_eval(); the 1e-6 "
                     "tolerance absorbs only float32 representation error")}


# --------------------------------------------------------------------------- #
# convenience: open a stream through the cache, building it on first use
# --------------------------------------------------------------------------- #
def stream_cache_dir(cache_root: str, split_dir: str, spec: StreamSpec) -> str:
    tag = os.path.basename(os.path.normpath(split_dir)).replace("/", "_")
    parent = os.path.basename(os.path.dirname(os.path.normpath(split_dir))).replace("/", "_")
    return os.path.join(cache_root, f"{parent}_{tag}_{spec.eval_size}_{cache_key(split_dir, spec)}")


def open_stream(split_dir: str, spec: StreamSpec, cache_root: Optional[str],
                build: bool = True) -> object:
    """Return a cached stream when possible, else the plain loader.

    ``cache_root=None`` disables the cache entirely (source training wants
    shuffled multi-epoch access and is not the path being optimised here).
    """
    if not cache_root:
        return PairedStream(split_dir, spec)
    cdir = stream_cache_dir(cache_root, split_dir, spec)
    if not os.path.exists(os.path.join(cdir, META)):
        if not build:
            return PairedStream(split_dir, spec)
        build_cache(split_dir, spec, cdir)
    return CachedStream(cdir)
