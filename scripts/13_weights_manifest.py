"""13 -- record the exact weights this study used, with SHA256 and provenance.

Spec §13 asks for "weights + SHA256".  Two classes of weights matter and they must
not be confused:

  theta_0 (source)      the pretrained MambaCue checkpoint for each source domain,
                        used unmodified as the frozen reference;
  adapter               the source-trained residual adapter (11,917 parameters)
                        that forms the actual starting point of each transfer
                        (zero-init at construction, then trained on source pairs);
  degrader              the source-fitted re-degradation model, used only as
                        evidence (never as a claim).

Provenance note that belongs in the report and not only here: the MambaCue
checkpoints are the authors' released weights.  Their exact training scope is not
documented in the files we have, so "zero-shot cross-domain" is what we can
claim operationally -- we never trained on the target -- not a verified statement
about the upstream training corpus.

Usage:
    python scripts/13_weights_manifest.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ROOT, adapter_ckpt_path, degrader_path, load_config,  # noqa: E402
                    resolve_ckpt, save_json)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entry(path: str, kind: str, **extra) -> dict:
    if not path or not os.path.exists(path):
        return {"kind": kind, "path": path, "exists": False, **extra}
    st = os.stat(path)
    return {"kind": kind, "path": path, "exists": True, "bytes": st.st_size,
            "sha256": sha256(path), **extra}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--backbone", default="mambacue")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    weights = {}
    for st in cfg["streams"]:
        try:
            ck = resolve_ckpt(cfg, st, args.backbone)
        except Exception as exc:  # missing checkpoint is a finding, not a crash
            ck = None
            weights.setdefault("errors", []).append(f"{st['name']}: {exc}")
        weights[f"{st['name']}:theta0"] = entry(ck, "source_checkpoint", stream=st["name"],
                                               source=st["source"], backbone=args.backbone)
        weights[f"{st['name']}:adapter"] = entry(
            adapter_ckpt_path(cfg, st["source"], args.backbone), "source_adapter",
            stream=st["name"], source=st["source"])
    for src in sorted({st["source"] for st in cfg["streams"]}):
        weights[f"degrader:{src}"] = entry(degrader_path(cfg, src, args.backbone),
                                          "redegradation_model", source=src)

    out_doc = {
        "backbone": args.backbone,
        "note": ("theta_0 for a transfer is the source adapter on top of the source "
                 "checkpoint, not the bare backbone. Provenance: released MambaCue "
                 "weights; upstream training corpus undocumented, so the claim is "
                 "operational zero-shot (no target training), not a verified statement "
                 "about upstream data."),
        "weights": weights,
    }
    out = args.out or os.path.join(ROOT, "docs", "weights_manifest.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_json(out_doc, out)
    n_ok = sum(1 for v in weights.values() if isinstance(v, dict) and v.get("exists"))
    print(f"[13] {n_ok} weights hashed -> {out}")
    for k, v in weights.items():
        if isinstance(v, dict) and v.get("exists"):
            print(f"     {k:32s} {v['bytes']/1e6:7.2f} MB  {v['sha256'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
