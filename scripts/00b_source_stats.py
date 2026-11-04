"""00b -- source-domain clean-image statistics (the TTA "prior").

A test-time update needs a target to move toward that does not require target
labels.  The only legitimate one in the strict single-source setting is a
*statistic of the source-domain clean images*, learned offline from the source
training pairs.  We store the per-channel mean/std in RGB plus the CIE-Lab a/b
distribution, and keep it small enough to inspect by eye.

This is not a target-domain statistic: nothing here reads target data.

Usage:
    python scripts/00b_source_stats.py --dataset LSUI --n 400
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, save_json, split_dir  # noqa: E402


sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from brctta.data import list_stems, load_rgb, resolve_path, to_tensor  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from skimage import color as skcolor

    cfg = load_config(args.config)
    root = split_dir(cfg, args.dataset, "train")
    stems = list_stems(root)
    step = max(1, len(stems) // args.n)
    picked = stems[::step][: args.n]

    rgb, lab = [], []
    for s in picked:
        im = load_rgb(resolve_path(root, s, "GT"))
        im = im.resize((256, 256))
        x = to_tensor(im).permute(1, 2, 0).numpy()  # H,W,3 in [0,1]
        rgb.append(x.reshape(-1, 3))
        lab.append(skcolor.rgb2lab(x).reshape(-1, 3))
    rgb = np.concatenate(rgb, 0)
    lab = np.concatenate(lab, 0)

    prior = {
        "dataset": args.dataset,
        "n_images": len(picked),
        "rgb_mean": rgb.mean(0).tolist(),
        "rgb_std": rgb.std(0).tolist(),
        "lab_mean": lab.mean(0).tolist(),
        "lab_std": lab.std(0).tolist(),
        "lab_ab_mean": lab[:, 1:].mean(0).tolist(),
        "lab_ab_std": lab[:, 1:].std(0).tolist(),
        "luma_mean": float(rgb.mean()),
        "saturation_mean": float((rgb.max(1) - rgb.min(1)).mean()),
        "source_split": os.path.join("/", *root.split(os.sep)[-3:]),
        "stems_used": len(picked),
    }
    out = args.out or os.path.join(cfg["paths"]["out_root"], "manifests", f"prior_{args.dataset}.json")
    save_json(prior, out)
    print(json.dumps({k: prior[k] for k in ("dataset", "n_images", "rgb_mean", "rgb_std",
                                            "lab_ab_mean", "saturation_mean")}, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
