#!/usr/bin/env python3
"""Freeze the runtime environment into docs/env.lock.json (spec §13).

WHY a *recorded* environment and not a lock file:
the pod has no conda binary on PATH -- `/home/user/Conda` is an envs directory
whose interpreters are addressed by absolute path.  Recreating the env from a
pip freeze is therefore the only thing that could silently drift: a `pip install`
inside CDF_FFT would change every number in the paper.  So we record what is
actually installed, plus the interpreter and the GPU, and hash the record.  Any
later run can be checked against it with `--check`.

Usage:
    bash scripts/gpu_run.sh 00f_env_lock.py            # write docs/env.lock.json
    bash scripts/gpu_run.sh 00f_env_lock.py --check    # verify, exit 3 on drift
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "env.lock.json")

# Only packages the project imports or that change numerics.  A full pip freeze
# is 300 lines of transitive noise; these are the ones that can move a PSNR.
TRACKED = [
    "torch", "torchvision", "numpy", "scipy", "Pillow", "opencv-python",
    "scikit-image", "scikit-learn", "pandas", "matplotlib", "timm",
    "mamba-ssm", "einops", "torchmetrics", "pyyaml", "tqdm",
]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pkg_versions() -> dict:
    from importlib import metadata

    out = {}
    for name in TRACKED:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def gpu_info() -> dict:
    info = {"cuda_available": None, "device_count": None, "name": None,
            "total_memory_bytes": None, "torch_cuda": None, "nvcc": None}
    try:
        import torch

        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["device_count"] = int(torch.cuda.device_count())
        if info["cuda_available"]:
            props = torch.cuda.get_device_properties(0)
            info["name"] = props.name
            info["total_memory_bytes"] = int(props.total_memory)
    except Exception as exc:  # noqa: BLE001 - recording a failure is the point
        info["error"] = repr(exc)
    try:
        info["nvcc"] = subprocess.run(["nvcc", "--version"], capture_output=True,
                                      text=True, timeout=20).stdout.strip().splitlines()[-1]
    except Exception:  # noqa: BLE001
        pass
    return info


def build() -> dict:
    record = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.node(),
        "packages": pkg_versions(),
        "gpu": gpu_info(),
        "torch_threads": os.environ.get("BRCTTA_TORCH_THREADS", "4"),
    }
    body = json.dumps(record, indent=2, sort_keys=True)
    record["record_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    return record


def main(argv: list[str]) -> int:
    check = "--check" in argv
    record = build()
    if check:
        if not os.path.exists(OUT):
            print(f"[env] no lock at {OUT}; run without --check first")
            return 3
        with open(OUT, encoding="utf-8") as fh:
            old = json.load(fh)
        diffs = []
        for key in ("python_version", "platform", "packages"):
            if old.get(key) != record.get(key):
                if key == "packages":
                    for pkg, ver in record["packages"].items():
                        if old.get("packages", {}).get(pkg) != ver:
                            diffs.append(f"  {pkg}: {old.get('packages', {}).get(pkg)} -> {ver}")
                else:
                    diffs.append(f"  {key}: {old.get(key)} -> {record.get(key)}")
        if diffs:
            print("[env] DRIFT detected:")
            print("\n".join(diffs))
            return 3
        print(f"[env] ok, matches {OUT}")
        return 0

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
    print(f"[env] wrote {OUT}")
    print(f"      python {record['python_version'].split()[0]}")
    print(f"      torch {record['packages'].get('torch')} cuda {record['gpu'].get('torch_cuda')} "
          f"device {record['gpu'].get('name')}")
    print(f"      record_sha256 {record['record_sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
