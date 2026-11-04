"""04b -- select the adapter step size on DEV, by the *value of gating*.

Why this criterion?  A test-time update is only interesting in the regime where
it sometimes helps and sometimes hurts.  If every candidate is a no-op the risk
controller has nothing to decide; if every candidate destroys the image the
answer is trivially "always keep" and the mechanism is untested.

The first version of this rule measured headroom against the *always accept*
arm, which is a trap: on UIEB2LSUI the 5x0.01 cell "won" with +0.46 dB of
image-wise headroom while always-accept had already collapsed aggregate PSNR
from 22.5 dB to 16.9 dB (mean per-image dPSNR stayed positive because a handful
of catastrophic images dominate the mean of dB values downwards without moving
the mean of the deltas).  Comparing against a collapsing reference rewards
destruction.

The rule below therefore compares the gate's achievable ceiling against the
**frozen** model -- the policy that never updates, which any gate can always
fall back to by rejecting everything:

    gain_vs_frozen = mean_i max(psnr_pre_i, psnr_post_i) - PSNR(frozen)

A cell is only worth using if a gate on its trajectory can beat doing nothing.
This is still a label-using diagnostic bound, never a method, and it is computed
on DEV only -- the test split is never touched here.

Usage:
    python scripts/04b_select_adapt.py --stream LSUI2UIEB --n 60 \
        --grid 1x0.001,3x0.003,5x0.01,8x0.02
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import adapter_ckpt_path, backbone_suffix, load_config, open_stream, resolve_ckpt, run_dir, save_json, split_dir  # noqa: E402


sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from brctta.adapt import AdaptConfig  # noqa: E402
from brctta.baselines import GatedAdaptation  # noqa: E402
from brctta.data import PairedStream, StreamSpec  # noqa: E402
from brctta.losses import load_prior  # noqa: E402
from brctta.model import build_adapted  # noqa: E402


def run_cell(ckpt, prior, stream, steps, lr, n, device, hidden, memory, gate, log_path=None,
             backbone="mambacue", adapter_ckpt=None):
    acfg = AdaptConfig(steps=steps, lr=lr, consistency=True, memory=memory)
    model = build_adapted(ckpt, device=device, hidden=hidden, backbone=backbone,
                          adapter_ckpt=adapter_ckpt)
    eng = GatedAdaptation(model, kind=gate, cfg=acfg, method=f"{gate}_{steps}x{lr}", prior=prior)
    t0 = time.time()
    s = eng.run(stream, limit=n, log_path=log_path)
    s["per_image_seconds"] = (time.time() - t0) / n
    return s


def run_frozen(ckpt, stream, n, device, hidden, backbone="mambacue", log_path=None,
               adapter_ckpt=None):
    """The 'never update' reference every gate can always fall back to."""
    from brctta.baselines import FrozenBaseline

    model = build_adapted(ckpt, device=device, hidden=hidden, backbone=backbone,
                          adapter_ckpt=adapter_ckpt)
    eng = FrozenBaseline(model, device=device)
    t0 = time.time()
    s = eng.run(stream, limit=n, log_path=log_path)
    s["per_image_seconds"] = (time.time() - t0) / n
    return s


def imagewise_bound(log_path):
    """Upper bound a gate could reach on the trajectories this cell actually visited.

    For every image the engine logged BOTH the score of the committed state
    (``psnr_pre``) and the score of the candidate (``psnr_post``, which for the
    always-accept arm is that candidate committed).  A gate can pick at most one
    of the two per image, so ``mean(max(pre, post))`` is a per-image upper bound
    on what any commit rule could have scored -- it needs the labels, so it is a
    diagnostic, not a method.  Unlike the greedy oracle it does not assume the
    trajectory stays fixed, so it is not distorted by myopia.
    """
    recs = [json.loads(l) for l in open(log_path, "r", encoding="utf-8") if l.strip()]
    if not recs:
        return float("nan")
    pre = np.array([r["psnr_pre"] for r in recs], dtype=np.float64)
    post = np.array([r["psnr_post"] for r in recs], dtype=np.float64)
    return float(np.maximum(pre, post).mean())


def worst10(v: np.ndarray, frac: float = 0.10) -> float:
    k = max(1, int(round(len(v) * frac)))
    return float(np.sort(v)[:k].mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--grid", default="1x0.001,3x0.003,5x0.01")
    ap.add_argument("--dev-domain", default="source")
    ap.add_argument("--split", default="auto")
    ap.add_argument("--memory", type=int, default=0)
    ap.add_argument("--backbone", default="mambacue", choices=["mambacue", "aqua"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    stream_cfg = next(s for s in cfg["streams"] if s["name"] == args.stream)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = stream_cfg["source"]
    if args.dev_domain == "source":
        probe_ds, probe_split = ds, "dev"
    elif args.dev_domain == "target":
        # target-domain DEV split: legitimate (dev != test) and the only dev set
        # that actually exhibits the shift we are calibrating for.
        probe_ds, probe_split = stream_cfg["target"], "dev"
    else:
        probe_ds, probe_split = args.dev_domain, "dev"

    split = args.split if args.split != "auto" else probe_split
    spec = StreamSpec(name=f"select_{args.stream}", source_dataset=probe_ds,
                      target_split=f"{probe_ds}/{split}", seed=stream_cfg["seed"])
    stream = open_stream(cfg, probe_ds, split, spec)
    print(f"selection dev domain: {probe_ds}/{split} ({len(stream)} images)", flush=True)

    ckpt = resolve_ckpt(cfg, stream_cfg, args.backbone)
    prior_path = os.path.join(cfg["paths"]["out_root"], "manifests", f"prior_{ds}.json")
    prior = load_prior(prior_path) if os.path.exists(prior_path) else None
    hidden = cfg["adapter"]["hidden"]

    out = run_dir(cfg, "selection", f"{args.stream}_{probe_ds}" + backbone_suffix(args.backbone))
    logdir = os.path.join(out, "logs")
    os.makedirs(logdir, exist_ok=True)

    adapter_ckpt = adapter_ckpt_path(cfg, ds, args.backbone)
    print(f"theta_0 = {'source_adapter' if adapter_ckpt else 'frozen_backbone'}", flush=True)
    frozen_log = os.path.join(logdir, "frozen.jsonl")
    frozen = run_frozen(ckpt, stream, args.n, device, hidden, args.backbone, frozen_log,
                        adapter_ckpt=adapter_ckpt)
    print(f"frozen: psnr={frozen['psnr_prequential']:.4f} "
          f"ssim={frozen['ssim_prequential']:.4f}", flush=True)

    rows = []
    for cell in args.grid.split(","):
        steps_s, lr_s = cell.split("x")
        steps, lr = int(steps_s), float(lr_s)
        row = {"cell": cell, "steps": steps, "lr": lr}
        for gate in ("always", "oracle"):
            lp = os.path.join(logdir, f"{gate}_{cell}.jsonl")
            s = run_cell(ckpt, prior, stream, steps, lr, args.n, device, hidden, args.memory,
                         gate, lp, backbone=args.backbone, adapter_ckpt=adapter_ckpt)
            row[gate] = {k: s[k] for k in ("psnr_prequential", "ssim_prequential", "dPSNR_mean",
                                           "NAR_0", "worst10_dPSNR", "accept_rate",
                                           "per_image_seconds")}
        row["headroom_dB"] = row["oracle"]["psnr_prequential"] - row["always"]["psnr_prequential"]
        row["worst10_improvement"] = row["oracle"]["worst10_dPSNR"] - row["always"]["worst10_dPSNR"]
        lp = os.path.join(logdir, f"always_{cell}.jsonl")
        recs = [json.loads(l) for l in open(lp, "r", encoding="utf-8") if l.strip()]
        d = np.array([r["psnr_post"] - r["psnr_pre"] for r in recs], dtype=np.float64)
        row["imagewise_bound_psnr"] = imagewise_bound(lp)
        row["imagewise_headroom_dB"] = row["imagewise_bound_psnr"] - row["always"]["psnr_prequential"]
        row["delta_sd_dB"] = float(d.std())
        row["delta_p90_dB"] = float(np.percentile(-d, 90))   # typical harm size
        row["dPSNR_worst10"] = worst10(d)
        row["frozen_psnr"] = frozen["psnr_prequential"]
        row["bound_vs_frozen_dB"] = row["imagewise_bound_psnr"] - frozen["psnr_prequential"]
        row["always_vs_frozen_dB"] = row["always"]["psnr_prequential"] - frozen["psnr_prequential"]
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("cell", "headroom_dB", "worst10_improvement")}), flush=True)
        print(f"  always: psnr={row['always']['psnr_prequential']:.4f} dPSNR={row['always']['dPSNR_mean']:+.4f} "
              f"NAR0={row['always']['NAR_0']:.3f} worst10={row['always']['worst10_dPSNR']:+.3f}", flush=True)
        print(f"  oracle: psnr={row['oracle']['psnr_prequential']:.4f} acc={row['oracle']['accept_rate']:.3f} "
              f"worst10={row['oracle']['worst10_dPSNR']:+.3f}", flush=True)
        print(f"  gateable: imagewise_bound={row['imagewise_bound_psnr']:.4f} "
              f"headroom={row['imagewise_headroom_dB']:+.4f} dB  sd(delta)={row['delta_sd_dB']:.3f} "
              f"p90harm={row['delta_p90_dB']:.3f} worst10={row['dPSNR_worst10']:+.3f}", flush=True)

    # Selection rule (pre-declared, applied identically to every stream).
    #
    # Rules tried and rejected, all for rewarding an adapter that wrecks images:
    #   * raw greedy-oracle headroom picked a cell whose ceiling was 0.6 dB BELOW
    #     the smallest-step ceiling;
    #   * maximising the greedy-oracle ceiling is distorted by oracle myopia;
    #   * image-wise headroom over always-accept let 5x0.01 win on UIEB2LSUI while
    #     always-accept collapsed aggregate PSNR by 5.7 dB.
    # Locked rule: among cells whose gateable ceiling beats simply freezing the
    # model by at least MIN_GAIN_VS_FROZEN, take the largest such gain.
    MIN_GAIN_VS_FROZEN = 0.05
    viable = [r for r in rows if r["bound_vs_frozen_dB"] >= MIN_GAIN_VS_FROZEN]
    if not viable:
        viable = rows
    best = max(viable, key=lambda r: (round(r["bound_vs_frozen_dB"], 4),
                                      -abs(r["always"]["dPSNR_mean"])))
    out = run_dir(cfg, "selection", f"{args.stream}_{probe_ds}")
    save_json({
        "stream": args.stream, "dev_domain": probe_ds, "split": split, "n": args.n,
        "n_images": len(stream),
        "criterion": ("max gain_vs_frozen = mean(max(psnr_pre,psnr_post)) - PSNR(frozen), "
                      "restricted to cells with gain >= 0.05 dB; dev split only. Rejected "
                      "rules: greedy-oracle headroom (rewards destruction), greedy-oracle "
                      "ceiling (oracle myopia), and image-wise headroom over always-accept "
                      "(rewards an arm that collapses aggregate PSNR)."),
        "min_gain_vs_frozen_dB": MIN_GAIN_VS_FROZEN,
        "frozen": {"psnr_prequential": frozen["psnr_prequential"],
                   "ssim_prequential": frozen["ssim_prequential"],
                   "dPSNR_mean": frozen.get("dPSNR_mean", 0.0)},
        "rows": rows, "selected": best["cell"], "selected_steps": best["steps"],
        "selected_lr": best["lr"], "selected_headroom_dB": best["imagewise_headroom_dB"],
        "selected_gain_vs_frozen_dB": best["bound_vs_frozen_dB"],
    }, os.path.join(out, "adapt_selection.json"))
    print(f"\nSELECTED {best['cell']} "
          f"(image-wise headroom over always {best['imagewise_headroom_dB']:+.4f} dB, "
          f"gain vs frozen {best['bound_vs_frozen_dB']:+.4f} dB)")
    print("wrote", os.path.join(out, "adapt_selection.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
