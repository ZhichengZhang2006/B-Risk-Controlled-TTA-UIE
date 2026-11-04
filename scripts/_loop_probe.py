"""Loop probe: where do the seconds go inside one adaptation/pretrain step?

The GPU probe (scripts/_time_probe.py) says one fwd+bwd on 8x256x256 costs
~0.2 s, yet a 32-image pretrain run took 59 s.  That is a 70x discrepancy, so
this measures the suspects separately, on the real cached stream:

    A. CachedStream.take() from an mmap over NFS  (random access page faults)
    B. CachedStream.take() with the array in RAM   (np.load without mmap)
    C. host->device copy
    D. forward only
    E. forward+backward
    F. full optimiser step

Usage: python scripts/_loop_probe.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import cache_root, load_config, resolve_ckpt, split_dir  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from brctta.cache import CachedStream, stream_cache_dir  # noqa: E402
from brctta.data import EVAL_SIZE, StreamSpec  # noqa: E402
from brctta.model import build_adapted  # noqa: E402


def timeit(fn, n=8):
    t0 = time.time()
    for _ in range(n):
        fn()
    return (time.time() - t0) / n


def main() -> int:
    cfg = load_config()
    sd = split_dir(cfg, "LSUI", "train")
    spec = StreamSpec(name="probe", source_dataset="LSUI", target_split="LSUI/train", seed=0)
    cdir = stream_cache_dir(cache_root(cfg), sd, spec)
    print("cache:", cdir, "exists:", os.path.isdir(cdir))

    t0 = time.time()
    cs_mm = CachedStream(cdir, mmap=True)
    print(f"A. open mmap              {time.time()-t0:.2f} s")
    print(f"A. take() mmap            {timeit(lambda: cs_mm.take(7), n=20)*1000:9.1f} ms/img")

    t0 = time.time()
    cs_ram = CachedStream(cdir, mmap=False)
    print(f"B. open into RAM          {time.time()-t0:.2f} s  "
          f"({cs_ram.x.nbytes/2**20:.0f} MiB x2)")
    print(f"B. take() RAM             {timeit(lambda: cs_ram.take(7), n=20)*1000:9.1f} ms/img")

    dev = "cuda"
    model = build_adapted(resolve_ckpt(cfg, {"source": "LSUI", "source_ckpt": "LSUI/best_model.pth"},
                                       "mambacue"), device=dev, hidden=32)
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.adapter_params(), lr=1e-3)

    idxs = list(range(8))
    xb = torch.stack([cs_ram.take(i)[0] for i in idxs])
    gtb = torch.stack([cs_ram.take(i)[1] for i in idxs])

    print(f"C. host->device (8)       {timeit(lambda: xb.to(dev), n=20)*1000:9.1f} ms")
    xd, gd = xb.to(dev), gtb.to(dev)
    print(f"D. forward (8)            {timeit(lambda: model(xd), n=5)*1000:9.1f} ms")
    with torch.no_grad():
        print(f"D'. forward no_grad (8)   {timeit(lambda: model(xd), n=5)*1000:9.1f} ms")
        print(f"D''. backbone only (8)    {timeit(lambda: model.backbone(xd), n=5)*1000:9.1f} ms")

    def fwd_bwd():
        y = model(xd)
        loss = (y - gd).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
    print(f"E. fwd+bwd (8)            {timeit(fwd_bwd, n=3)*1000:9.1f} ms")

    def full_step():
        y = model(xd)
        loss = (y - gd).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.adapter_params(), 1.0)
        opt.step()
    print(f"F. full step (8)          {timeit(full_step, n=3)*1000:9.1f} ms")

    def ram_batch():
        xs = torch.stack([cs_ram.take(i)[0] for i in idxs])
        ys = torch.stack([cs_ram.take(i)[1] for i in idxs])
        return xs.to(dev), ys.to(dev)
    print(f"G. take+stack+copy (8)    {timeit(ram_batch, n=5)*1000:9.1f} ms")

    def mm_batch():
        xs = torch.stack([cs_mm.take(i)[0] for i in idxs])
        return xs.to(dev)
    print(f"H. mmap take+stack (8)    {timeit(mm_batch, n=5)*1000:9.1f} ms")

    print("\nprojected full epoch (3423 imgs, 428 steps of 8):")
    print(f"  GPU fwd+bwd only        {428*timeit(fwd_bwd, n=2):.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
