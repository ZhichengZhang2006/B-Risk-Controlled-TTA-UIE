"""Micro-probe of CachedStream.take(): which numpy/torch op costs 185 ms?

_loop_probe.py reported 185 ms for one `take()` of a 256x256x3 uint8 array --
three orders of magnitude more than the arithmetic implies.  The suspects are
thread oversubscription (this pod has 96 CPUs and 16 GB RAM, so torch's default
intra-op pool can thrash) and the HWC->CHW contiguous copy.  This isolates each
stage and repeats the whole thing with torch.set_num_threads(1).
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import cache_root, load_config, split_dir  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from brctta.cache import CachedStream, stream_cache_dir  # noqa: E402
from brctta.data import StreamSpec  # noqa: E402


def t(fn, n=20):
    t0 = time.time()
    for _ in range(n):
        fn()
    return (time.time() - t0) / n * 1000


def main() -> int:
    cfg = load_config()
    sd = split_dir(cfg, "LSUI", "train")
    spec = StreamSpec(name="probe", source_dataset="LSUI", target_split="LSUI/train", seed=0)
    cs = CachedStream(stream_cache_dir(cache_root(cfg), sd, spec), mmap=False)
    arr = cs.x[7]
    print("array", arr.shape, arr.dtype, arr.nbytes, "bytes")
    print("torch threads:", torch.get_num_threads(), "interop:", torch.get_num_interop_threads())
    try:
        print("omp threads:", os.environ.get("OMP_NUM_THREADS"))
    except Exception:
        pass

    print("\n-- default thread settings")
    print(f"  arr.copy()                 {t(lambda: arr.copy()):8.3f} ms")
    npc = lambda: np.array(arr, dtype=np.uint8, copy=True)
    print(f"  np.array copy              {t(npc):8.3f} ms")
    print(f"  from_numpy                 {t(lambda: torch.from_numpy(npc())):8.3f} ms")
    ten = torch.from_numpy(npc())
    print(f"  permute+contiguous         {t(lambda: ten.permute(2,0,1).contiguous()):8.3f} ms")
    print(f"  float()+div_               {t(lambda: ten.permute(2,0,1).contiguous().to(torch.float32).div_(255.0)):8.3f} ms")
    print(f"  full take()                {t(lambda: cs.take(7)):8.3f} ms")

    n = torch.get_num_threads()
    torch.set_num_threads(1)
    print(f"\n-- torch.set_num_threads(1)  (was {n})")
    print(f"  permute+contiguous         {t(lambda: ten.permute(2,0,1).contiguous()):8.3f} ms")
    print(f"  float()+div_               {t(lambda: ten.permute(2,0,1).contiguous().to(torch.float32).div_(255.0)):8.3f} ms")
    print(f"  full take()                {t(lambda: cs.take(7)):8.3f} ms")
    print(f"  stack 8 takes              {t(lambda: torch.stack([cs.take(i)[0] for i in range(8)]), n=5):8.3f} ms")

    print("\n-- raw numpy path, no torch at all (uint8 HWC -> CHW float32)")
    print(f"  np.transpose+astype        {t(lambda: np.asarray(arr, dtype=np.float32).transpose(2,0,1)/255.0):8.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
