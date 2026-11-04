"""Data-path probe: is the training step slow because of I/O, not the GPU?

The GPU probe says one fwd+bwd on 8x256x256 costs ~207 ms.  The pretrain script
measured ~11.5 s per batch of 8, so ~11.3 s is spent outside the GPU.  That is a
data-path problem, and it matters far beyond the pretrain: every episodic
evaluation touches the same loader.

This measures, separately:
  * raw file open+decode (PIL) for the original-size input and GT,
  * the whole-image bicubic resize to 256x256,
  * the tensor conversion,
  * the same for a second, warm read (NFS page cache),
  * and how much of that a small LRU cache removes.

Usage: python scripts/_data_probe.py [--dataset LSUI]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, split_dir  # noqa: E402


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from brctta.data import PairedStream, StreamSpec, list_stems, resolve_path, to_tensor  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="LSUI")
    ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()
    cfg = load_config()
    root = split_dir(cfg, args.dataset, "train")
    stems = list_stems(root)[: args.n]

    t = {"open": 0.0, "resize": 0.0, "tensor": 0.0, "bytes": 0}
    sizes = []
    for s in stems:
        for sub in ("input", "GT"):
            p = resolve_path(root, s, sub)
            t0 = time.time()
            im = Image.open(p)
            im.load()
            t1 = time.time()
            t["open"] += t1 - t0
            t["bytes"] += os.path.getsize(p)
            sizes.append(im.size)
            r = im.convert("RGB").resize((256, 256), Image.BICUBIC, reducing_gap=3.0)
            t2 = time.time()
            t["resize"] += t2 - t1
            _ = to_tensor(r)
            t["tensor"] += time.time() - t2

    n_img = 2 * len(stems)
    print(f"{args.dataset}: {n_img} images from {root}")
    print(f"  avg original size      {np.mean([w for w, h in sizes]):.0f} x {np.mean([h for w, h in sizes]):.0f}")
    print(f"  total bytes            {t['bytes']/2**20:.1f} MiB")
    print(f"  open+decode            {t['open']/n_img*1000:7.1f} ms/img")
    print(f"  resize 256             {t['resize']/n_img*1000:7.1f} ms/img")
    print(f"  to tensor              {t['tensor']/n_img*1000:7.1f} ms/img")
    print(f"  per image (cold)       {(t['open']+t['resize']+t['tensor'])/n_img*1000:7.1f} ms")

    # warm re-read: same files again, page cache should hold them
    t0 = time.time()
    for s in stems[:8]:
        for sub in ("input", "GT"):
            im = Image.open(resolve_path(root, s, sub))
            im.load()
    warm = (time.time() - t0) / 16
    print(f"  per image (warm)       {warm*1000:7.1f} ms")

    # what the loader actually costs through PairedStream
    spec = StreamSpec(name=f"probe_{args.dataset}", source_dataset=args.dataset,
                      target_split=f"{args.dataset}/train", seed=0)
    ds = PairedStream(root, spec)
    t0 = time.time()
    for i in range(min(8, len(ds))):
        ds.take(i)
    print(f"  PairedStream.take      {(time.time()-t0)/min(8, len(ds))*1000:7.1f} ms/img")
    print("\nimplied batch of 8 (cold) =",
          f"{(t['open']+t['resize']+t['tensor'])/n_img*16:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
