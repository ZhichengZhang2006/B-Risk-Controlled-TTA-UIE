"""Timing probe: where does the training step time go on this shared pod?

The first pretrain run took 92 s for 64 images (8 batches), i.e. ~11.5 s/batch.
That is far too slow for a 2.4M-parameter model at 256x256, so before spending
GPU-hours on a longer run this isolates the cost:

  backbone fwd/bwd, adapter fwd/bwd, image loading, and the effect of
  cudnn.benchmark / channels_last / autocast.

Also reports the GPU's current utilisation so contention from other tenants is
visible in the record.

Usage: python scripts/_time_probe.py
"""

from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, resolve_ckpt  # noqa: E402


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from brctta.model import build_adapted  # noqa: E402


def bench(fn, n=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n


def main() -> int:
    cfg = load_config()
    dev = "cuda"
    print("gpu:", torch.cuda.get_device_name(0))
    print("mem allocated/free:",
          f"{torch.cuda.memory_allocated()/2**30:.2f} / "
          f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved())/2**30:.2f} GiB")

    model = build_adapted(resolve_ckpt(cfg, {"source": "LSUI", "source_ckpt": "LSUI/best_model.pth"}, "mambacue"),
                          device=dev, hidden=32)
    for p in model.backbone.parameters():
        p.requires_grad_(False)

    xb = torch.rand(8, 3, 256, 256, device=dev)
    gtb = torch.rand(8, 3, 256, 256, device=dev)

    print("\n-- default flags")
    print(f"  backbone fwd (no_grad)  {bench(lambda: model.backbone(xb))*1000:7.1f} ms / 8 imgs")
    print(f"  full fwd                {bench(lambda: model(xb))*1000:7.1f} ms / 8 imgs")

    def step():
        y = model(xb)
        loss = (y - gtb).abs().mean()
        model.zero_grad(set_to_none=True)
        loss.backward()
    print(f"  fwd+bwd                 {bench(step, n=3)*1000:7.1f} ms / 8 imgs")

    torch.backends.cudnn.benchmark = True
    print("\n-- cudnn.benchmark=True")
    print(f"  backbone fwd (no_grad)  {bench(lambda: model.backbone(xb))*1000:7.1f} ms")
    print(f"  fwd+bwd                 {bench(step, n=3)*1000:7.1f} ms")

    print("\n-- autocast bf16")
    def step_bf16():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = model(xb)
            loss = (y.float() - gtb).abs().mean()
        model.zero_grad(set_to_none=True)
        loss.backward()
    print(f"  fwd+bwd                 {bench(step_bf16, n=3)*1000:7.1f} ms")

    print("\n-- batch 16 at default flags")
    xb2 = torch.rand(16, 3, 256, 256, device=dev)
    gt2 = torch.rand(16, 3, 256, 256, device=dev)
    def step16():
        y = model(xb2)
        loss = (y - gt2).abs().mean()
        model.zero_grad(set_to_none=True)
        loss.backward()
    print(f"  fwd+bwd                 {bench(step16, n=3)*1000:7.1f} ms / 16 imgs")

    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        print("\nnvidia-smi:", out.stdout.strip())
    except Exception as exc:  # pragma: no cover
        print("nvidia-smi failed:", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
