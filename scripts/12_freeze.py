#!/usr/bin/env python3
"""Freeze the evaluation surface (spec §13: freeze eval + acceptance hashes).

The spec forbids, after seeing test results, editing the metric, subsetting the
test images, shortening the stream or re-picking seeds.  A hash manifest cannot
prove intent, but it makes any such edit *visible*: `--check` recomputes every
digest and lists the files whose bytes changed.  It is run by `08_acceptance.py`
so the acceptance report carries the verdict, and it should be run again when
the repo is handed over.

What is frozen:
  * the metric implementation            src/brctta/metrics.py
  * the stream definition / resampling   src/brctta/data.py, src/brctta/cache.py
  * the protocol engine                  src/brctta/adapt.py, signals.py
  * the acceptance gate itself           scripts/08_acceptance.py
  * the comparison-set builder           scripts/08_comparison.py
  * the pinned SOTA references           configs/sota_reference.json
  * the data manifests, if produced      runs/manifests/*.json

NOT frozen (deliberately): anything that only *describes* a run (README, plots,
ledger) and anything fitted on dev data (risk controller, thresholds, adapter
weights) -- those must change as the research proceeds, and their provenance is
recorded in the experiment ledger instead.

Usage:
    bash scripts/gpu_run.sh 12_freeze.py           # write docs/frozen_hashes.json
    bash scripts/gpu_run.sh 12_freeze.py --check   # verify, exit 4 on drift
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "frozen_hashes.json")

FROZEN = [
    "src/brctta/metrics.py",
    "src/brctta/data.py",
    "src/brctta/cache.py",
    "src/brctta/adapt.py",
    "src/brctta/signals.py",
    "src/brctta/risk.py",
    "src/brctta/baselines.py",
    "scripts/08_acceptance.py",
    "scripts/08_comparison.py",
    "scripts/05_online_eval.py",
    "configs/sota_reference.json",
    "configs/main.yaml",
]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect() -> dict:
    hashes, missing = {}, []
    for rel in FROZEN:
        path = os.path.join(ROOT, rel)
        if os.path.exists(path):
            hashes[rel] = sha256_file(path)
        else:
            missing.append(rel)
    manifests = {}
    mdir = os.path.join(ROOT, "runs", "manifests")
    if os.path.isdir(mdir):
        for name in sorted(os.listdir(mdir)):
            if name.endswith(".json"):
                manifests[f"runs/manifests/{name}"] = sha256_file(os.path.join(mdir, name))
    return {"frozen": hashes, "manifests": manifests, "missing": missing}


def main(argv: list[str]) -> int:
    check = "--check" in argv
    cur = collect()
    if check:
        if not os.path.exists(OUT):
            print(f"[freeze] no manifest at {OUT}; run without --check first")
            return 4
        with open(OUT, encoding="utf-8") as fh:
            old = json.load(fh)
        drift = [rel for rel, digest in old.get("frozen", {}).items()
                 if cur["frozen"].get(rel) != digest]
        added = [rel for rel in cur["frozen"] if rel not in old.get("frozen", {})]
        if drift or added:
            print("[freeze] EVALUATION SURFACE CHANGED:")
            for rel in drift:
                print(f"  modified: {rel}")
            for rel in added:
                print(f"  added:    {rel}")
            print("  If this was not intended, restore the file.  If it was, the")
            print("  acceptance report must state that results were produced before")
            print("  the change and are therefore not covered by it.")
            return 4
        print(f"[freeze] ok, {len(cur['frozen'])} files match {OUT}")
        return 0

    body = json.dumps({"frozen": cur["frozen"], "manifests": cur["manifests"]},
                      indent=2, sort_keys=True)
    payload = json.loads(body)
    payload["manifest_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[freeze] wrote {OUT} ({len(payload['frozen'])} frozen files, "
          f"{len(payload['manifests'])} manifests)")
    if cur["missing"]:
        print(f"[freeze] warning: missing {cur['missing']}")
    print(f"[freeze] manifest_sha256 {payload['manifest_sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
