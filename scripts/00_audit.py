"""00 -- data audit + locked manifests.

Writes, for every split used by the locked protocol:
  * pair counts, order hash, probe sizes,
  * optional per-file SHA256 (slow on NFS, so it is opt-in),
  * the exact resampling rule.

Also checks the trap the spec warns about: whether UFO-120 pairs are
super-resolution pairs (input != GT size).  We *report* it and keep the
whole-image 256x256 rule, which neutralises the SR task; we never claim SR.

Usage:
    python scripts/00_audit.py --config configs/main.yaml [--hash]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_config, save_json, split_dir  # noqa: E402


sys.path.insert(0, os.path.join(ROOT, "src"))
from brctta.data import StreamSpec, build_manifest, list_stems, load_rgb, resolve_path  # noqa: E402


def audit_split(name: str, path: str, size: int) -> dict:
    stems = list_stems(path)
    sizes_in, sizes_gt = {}, {}
    sr_pairs = 0
    for s in stems:
        wi, hi = load_rgb(resolve_path(path, s, "input")).size
        wg, hg = load_rgb(resolve_path(path, s, "GT")).size
        sizes_in[f"{wi}x{hi}"] = sizes_in.get(f"{wi}x{hi}", 0) + 1
        sizes_gt[f"{wg}x{hg}"] = sizes_gt.get(f"{wg}x{hg}", 0) + 1
        if (wi, hi) != (wg, hg):
            sr_pairs += 1

    spec = StreamSpec(name=name, source_dataset=name.split("/")[0], target_split=name, eval_size=size)
    return {
        "split": name,
        "path": path,
        "num_pairs": len(stems),
        "input_size_histogram": dict(sorted(sizes_in.items(), key=lambda kv: -kv[1])[:6]),
        "gt_size_histogram": dict(sorted(sizes_gt.items(), key=lambda kv: -kv[1])[:6]),
        "size_mismatch_pairs": sr_pairs,
        "sr_pair_fraction": sr_pairs / max(len(stems), 1),
        "stream_spec": spec.__dict__,
        "stream_digest": spec.digest(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--hash", action="store_true", help="also compute per-file SHA256 (slow)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = args.out or os.path.join(cfg["paths"]["out_root"], "manifests")
    os.makedirs(out, exist_ok=True)

    report = {"data_root": cfg["paths"]["data_root"], "eval_size": cfg["eval"]["size"], "splits": {}}
    for ds, splits in cfg["splits"].items():
        for split, rel in splits.items():
            key = f"{ds}/{split}"
            path = os.path.join(cfg["paths"]["data_root"], rel)
            info = audit_split(key, path, cfg["eval"]["size"])
            if args.hash:
                spec = StreamSpec(name=key, source_dataset=ds, target_split=key, eval_size=cfg["eval"]["size"])
                info["manifest"] = build_manifest(path, spec, verify_hash=True)
            report["splits"][key] = info
            print(f"{key:<16} pairs={info['num_pairs']:<5} mismatch={info['size_mismatch_pairs']:<4} "
                  f"in={list(info['input_size_histogram'])[:2]} gt={list(info['gt_size_histogram'])[:2]}")

    # sanity: the three locked transfers must be present with paired files
    for st in cfg["streams"]:
        tgt = cfg["splits"][st["target"]]["test"]
        p = os.path.join(cfg["paths"]["data_root"], tgt)
        n = len(list_stems(p))
        report.setdefault("streams", {})[st["name"]] = {
            "source": st["source"], "target": st["target"], "target_split": tgt,
            "target_test_images": n, "seed": st["seed"], "source_ckpt": st["source_ckpt"],
        }
        print(f"stream {st['name']:<12} source={st['source']:<5} target_test={n}")

    save_json(report, os.path.join(out, "data_audit.json"))
    print("wrote", os.path.join(out, "data_audit.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
