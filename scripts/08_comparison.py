"""08a -- build the measured comparison set that §11's gate is evaluated against.

Why this is not just "read the paper numbers": §11 defines comparable SOTA as

    the strongest *available* method on the same transfer, with the same source
    data / pretraining permission, the same (absent) target-label information and
    the same online protocol.

Target-supervised UIE SOTA (a model trained on the target's paired data) is
expressly *not* the fair main gate -- it is a separate "target supervision
reference" that only supplies an absolute coordinate.  Comparing an unsupervised
TTA number against a fully supervised in-domain number would fail by 15-20% for
every method in this field, including the published TTA papers.

So the main gate is built from arms this project actually ran, under one frozen
protocol (256x256 whole-image resize, same stems, same order, same seeds, same
metric code), across two backbones:

  source_frozen   no update, BARE backbone              (weak frozen reference)
  source_adapter  no update, backbone + source-trained residual = theta_0
                                                        (the honest "do nothing")
  strong_frozen   no update, AquaClarity                (§11: a *strong* frozen)
  always_accept   vanilla TTA                           (strong TTA reference)
  simple_gate     one scalar + threshold                (§11: simple reject rule)
  stochastic      CoTTA-style restoration               (strong TTA reference)
  random_gate     chance control                        (sanity only)
  ours            the risk-controlled gate              (this work)
  oracle          label-using diagnostic                (NEVER a gate competitor)

``oracle`` is excluded from the gate on principle: it reads the ground truth, so
it is not a deployable method and cannot be called SOTA under an unsupervised
permission.  It is reported in the diagnostic table only.

Usage:
    python scripts/08_comparison.py
    python scripts/08_comparison.py --include-tag smoke   # ignore *_smoke runs
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_config, run_dir, save_json  # noqa: E402


# Arms that may be called "comparable SOTA" (deployable under the unsupervised,
# source-only permission of the locked protocol).
GATE_ARMS = ("source_frozen", "source_adapter", "strong_frozen", "always_accept",
             "simple_gate", "stochastic", "random_gate")
# Arms shown but never eligible: label-using or non-deployable.
DIAGNOSTIC_ARMS = ("oracle",)
OURS = "ours"

METRICS = ("psnr", "ssim", "mse")


def read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def collect(cfg):
    """Scan every produced result file and index it by (stream, arm, variant)."""
    out_root = cfg["paths"]["out_root"]
    found = []

    for path in sorted(glob.glob(os.path.join(out_root, "*", "online_results.json"))):
        d = os.path.dirname(path)
        payload = read_json(path)
        stream = payload.get("stream") or os.path.basename(d).split("_")[0]
        backbone = payload.get("backbone") or ("aqua" if d.endswith("_aqua") else "mambacue")
        variant = os.path.basename(d)
        # only primary runs (no _smoke / _dbg suffixes) may enter the gate
        if any(k in variant for k in ("_smoke", "_dbg", "_test", "_tmp")):
            continue
        for m in payload.get("methods", []):
            found.append({
                "stream": stream, "arm": m["method"], "variant": variant,
                "backbone": backbone, "n": payload.get("n"),
                "order_digest": payload.get("order_digest"),
                "source": path,
                "psnr": m.get("psnr_prequential"), "ssim": m.get("ssim_prequential"),
                "mse": m.get("mse_prequential"),
                "dPSNR_mean": m.get("dPSNR_mean"), "dSSIM_mean": m.get("dSSIM_mean"),
                "NAR_0": m.get("NAR_0"), "NAR_0.5dB": m.get("NAR_0.5dB"),
                "worst10_dPSNR": m.get("worst10_dPSNR"),
                "accept_rate": m.get("accept_rate"), "keep_rate": m.get("keep_rate"),
                "reset_rate": m.get("reset_rate"),
                "false_accept_rate": m.get("false_accept_rate"),
                "harmful_update_accept_ratio": m.get("harmful_update_accept_ratio"),
                "beneficial_false_reject_rate": m.get("beneficial_false_reject_rate"),
                "benefit_retention": m.get("benefit_retention"),
                "regret_dB_per_image": m.get("regret_dB_per_image"),
            })

    for path in sorted(glob.glob(os.path.join(out_root, "strong_frozen_*", "*.json"))):
        payload = read_json(path)
        if any(k in path for k in ("_smoke", "_dbg", "_tmp")):
            continue
        role = payload.get("role", "cross_domain")
        ds = payload.get("eval_dataset")
        split = payload.get("split", "test")
        if role != "cross_domain":
            continue   # in-domain checkpoints have seen target labels: never gated
        # map dataset -> the stream whose target it is
        stream = None
        for st in cfg["streams"]:
            if st["target"] == ds:
                stream = st["name"]
        if stream is None:
            continue
        found.append({
            "stream": stream, "arm": "strong_frozen", "role": role,
            "variant": os.path.basename(os.path.dirname(path)),
            "backbone": payload.get("backbone", "aqua"), "n": payload.get("n"),
            "order_digest": payload.get("order_digest"), "source": path,
            "psnr": payload.get("psnr"), "ssim": payload.get("ssim"), "mse": payload.get("mse"),
            "dPSNR_mean": 0.0, "dSSIM_mean": 0.0, "NAR_0": 0.0, "NAR_0.5dB": 0.0,
            "worst10_dPSNR": 0.0, "accept_rate": 0.0, "keep_rate": 1.0, "reset_rate": 0.0,
            "false_accept_rate": None, "harmful_update_accept_ratio": None,
            "beneficial_false_reject_rate": None, "benefit_retention": None,
            "regret_dB_per_image": None,
        })
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    arms = collect(cfg)
    streams = {}

    # the target-supervision reference is reported, never gated on
    ts_ref = read_json(os.path.join(ROOT, "configs", "sota_reference.json")) or {}
    ts_ref.pop("_meta", None)

    for st in cfg["streams"]:
        name = st["name"]
        rows = [a for a in arms if a["stream"] == name]
        if not rows:
            continue
        seen = {}
        for a in rows:
            seen.setdefault(a["arm"], []).append(a)
        # prefer the mambacue variant for "ours" (the model this project owns),
        # but keep every measured arm visible
        gate_rows = [a for a in rows if a["arm"] in GATE_ARMS]
        best_psnr = max(gate_rows, key=lambda a: a["psnr"]) if gate_rows else None
        best_ssim = max(gate_rows, key=lambda a: a["ssim"]) if gate_rows else None
        ours = [a for a in rows if a["arm"] == OURS]
        streams[name] = {
            "target": st["target"],
            "n_arms": len(rows),
            "arms": sorted(rows, key=lambda a: -a["psnr"]),
            "best_gate_psnr": best_psnr,
            "best_gate_ssim": best_ssim,
            "ours": ours,
            "target_supervision_reference": ts_ref.get(name),
            "diagnostic_arms": [a for a in rows if a["arm"] in DIAGNOSTIC_ARMS],
        }

    payload = {
        "_meta": {
            "definition": ("S_m = best measured arm on the same transfer under the same "
                           "protocol and the same unsupervised permission, chosen from "
                           "deployable arms only (never the label-using oracle)."),
            "gate_arms": list(GATE_ARMS),
            "diagnostic_arms_excluded_from_gate": list(DIAGNOSTIC_ARMS),
            "target_supervision_reference_status": ("reported separately as an absolute "
                                                    "coordinate; NOT part of the §11 gate"),
            "protocol": "256x256 whole-image BICUBIC resize, no crop/pad, same stems, same order",
        },
        "streams": streams,
    }
    out = os.path.join(run_dir(cfg, "comparison"), "comparison_set.json")
    save_json(payload, out)

    for name, s in streams.items():
        print(f"\n== {name} (target {s['target']}, {s['n_arms']} arm-results)")
        for a in s["arms"]:
            mark = " *GATE-BEST*" if (s["best_gate_psnr"] and a is s["best_gate_psnr"]) else ""
            print(f"   {a['arm']:<14} {a['backbone']:<9} n={a['n']:<4} "
                  f"PSNR {a['psnr']:.4f}  SSIM {a['ssim']:.4f}{mark}")
        if s["target_supervision_reference"]:
            r = s["target_supervision_reference"]
            print(f"   [target-sup ref, not gated] {r.get('method', '')} "
                  f"PSNR {r.get('psnr')} SSIM {r.get('ssim')}")
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
