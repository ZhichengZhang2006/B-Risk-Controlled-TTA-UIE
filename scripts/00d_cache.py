"""00d -- materialise the stream caches and prove they are lossless.

Measured motivation (scripts/_data_probe.py, LSUI/train):

    open+decode 131 ms/img | resize 256 62 ms/img | to tensor 109 ms/img
    cold 302 ms/img, warm 24 ms/img, PairedStream 412 ms/img

A batch of 8 therefore burns ~4.8 s of host time against ~0.21 s of GPU on a
shared pod with 16 GB RAM over NFS.  The cache stores exactly the tensor the
protocol produces, as uint8, and `verify_against_loader` re-derives a random
sample through the slow path and requires bitwise agreement -- so the cache is
a memoisation of §5.1, not a different protocol.

Usage:
    python scripts/00d_cache.py --splits all
    python scripts/00d_cache.py --splits LSUI/train,UIEB/test --verify 8
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import cache_root, load_config, open_stream, save_json, split_dir  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from brctta.cache import CachedStream, build_cache, cache_key, stream_cache_dir, verify_against_loader  # noqa: E402
from brctta.data import EVAL_SIZE, StreamSpec  # noqa: E402


def parse_splits(cfg, spec_arg: str):
    out = []
    for ds in cfg["splits"]:
        for split in cfg["splits"][ds]:
            key = f"{ds}/{split}"
            if spec_arg in ("all", "") or key in spec_arg.split(","):
                out.append(key)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--splits", default="all", help="'all' or comma list like LSUI/train,UIEB/test")
    ap.add_argument("--verify", type=int, default=5, help="items re-derived through the slow path")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = cache_root(cfg)
    os.makedirs(root, exist_ok=True)
    print(f"cache root {root}")

    report = {"cache_root": root, "eval_size": EVAL_SIZE, "splits": {}}
    total_bytes = 0
    for key in parse_splits(cfg, args.splits):
        ds, split = key.split("/")
        sd = split_dir(cfg, ds, split)
        spec = StreamSpec(name=f"cache_{key.replace('/', '_')}", source_dataset=ds,
                          target_split=key, seed=0, eval_size=EVAL_SIZE)
        cdir = stream_cache_dir(root, sd, spec)
        t0 = time.time()
        build_cache(sd, spec, cdir, overwrite=args.overwrite)
        build_s = time.time() - t0
        cs = CachedStream(cdir)
        v = verify_against_loader(cdir, sd, spec, n=args.verify) if args.verify > 0 else {"ok": None}
        size = os.path.getsize(os.path.join(cdir, "x.npy")) + os.path.getsize(os.path.join(cdir, "y.npy"))
        total_bytes += size
        report["splits"][key] = {
            "dir": cdir, "key": cache_key(sd, spec), "n": len(cs),
            "eval_size": cs.eval_size, "bytes": size,
            "build_seconds": round(build_s, 1),
            "meta": cs.meta.get("sha256_x"), "verify": v,
        }
        print(f"  {key:16s} n={len(cs):5d} {size/2**20:7.1f} MiB  build={build_s:6.1f}s  "
              f"verify_ok={v['ok']}  max|dx|={v.get('max_abs_dx')}")
        if v["ok"] is False:
            print("  [FATAL] cache is not a faithful copy of the protocol tensor")
            save_json(report, os.path.join(cfg["paths"]["out_root"], "manifests", "cache_report.json"))
            return 2

    report["total_bytes"] = total_bytes
    save_json(report, os.path.join(cfg["paths"]["out_root"], "manifests", "cache_report.json"))
    print(f"\ntotal {total_bytes/2**30:.2f} GiB cached; report written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
