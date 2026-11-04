"""04f -- is the self-supervised objective pointing the right way?

Motivation: 04e showed the adapter can gain +1.45 dB on UIEB dev under a
label loss, while the project's self-supervised objective *loses* 0.32 dB.  Two
explanations remain:

  (i)  scale problem  -- the gradient direction is right but the step size is
       wrong; a line search along the self-supervised step should then still gain.
  (ii) signal problem -- the two gradients disagree in direction, so no step size
       can fix it and the objective itself has to change.

This script measures, on DEV images only:

  cos(g_ss, g_sup)          cosine similarity of the two gradients at theta_0
  cos(g_ss, g_sup_pc)       same after projecting out the component of g_sup the
                            adapter cannot represent (sanity: must stay positive)
  line_search               true dPSNR when stepping along +g_ss at several
                            step scales alpha, and along -g_sup (descent)
  best_alpha                the alpha maximising true dPSNR along the ss direction

A consistently positive cosine plus a usable best_alpha means the objective is
merely mis-scaled.  A cosine near zero or negative means the objective has to be
replaced -- which is exactly the case spec §4 warns about ("a shrinking unlabeled
consistency loss is not evidence of a real gain").

DEV ONLY.

Usage:
    python scripts/04f_grad_align.py --stream LSUI2UIEB --n 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import adapter_ckpt_path, backbone_suffix, load_config, open_stream, resolve_ckpt, run_dir, save_json, split_dir  # noqa: E402


sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from brctta import metrics as M  # noqa: E402
from brctta.data import PairedStream, StreamSpec  # noqa: E402
from brctta.losses import load_prior, self_supervised_loss  # noqa: E402
from brctta.model import build_adapted  # noqa: E402


def supervised_loss(y: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    l1 = (y - gt).abs().mean()
    a = y - y.mean(dim=(2, 3), keepdim=True)
    b = gt - gt.mean(dim=(2, 3), keepdim=True)
    ncc = (a * b).sum(dim=(2, 3)) / (a.norm(dim=(2, 3)) * b.norm(dim=(2, 3)) + 1e-6)
    return l1 + 0.2 * (1.0 - ncc.mean())


def flat_grad(model) -> torch.Tensor:
    return torch.cat([p.grad.reshape(-1) for p in model.adapter_params() if p.grad is not None]).detach()


def apply_flat(model, theta0, vec: torch.Tensor, alpha: float = 1.0) -> None:
    """theta <- theta0 + alpha * vec, where vec is a *unit* direction.

    A previous version multiplied the raw gradient, which made alpha a multiple of
    |g| and produced nonsense (a "supervised descent step" of -14 dB).  The
    direction is normalised here, so alpha is a step length in parameter space and
    is comparable across images regardless of gradient magnitude.
    """
    model.restore(theta0)
    norm = float(vec.norm())
    if norm < 1e-12:
        return
    step = (vec / norm) * alpha
    offset = 0
    with torch.no_grad():
        for p in model.adapter_params():
            n = p.numel()
            p.add_(step[offset:offset + n].reshape_as(p))
            offset += n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--dev-domain", default="target")
    ap.add_argument("--backbone", default="mambacue", choices=["mambacue", "aqua"])
    # step lengths in parameter space (the direction is unit-normalised)
    ap.add_argument("--alphas", default="0.001,0.005,0.01,0.05,0.1,0.5,1.0")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sc = next(s for s in cfg["streams"] if s["name"] == args.stream)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    probe_ds = sc["target"] if args.dev_domain == "target" else sc["source"]

    spec = StreamSpec(name=f"align_{args.stream}", source_dataset=probe_ds,
                      target_split=f"{probe_ds}/dev", seed=sc["seed"])
    stream = open_stream(cfg, probe_ds, "dev", spec)
    n = min(args.n, len(stream))
    prior = load_prior(os.path.join(cfg["paths"]["out_root"], "manifests", f"prior_{sc['source']}.json"))

    model = build_adapted(resolve_ckpt(cfg, sc, args.backbone), device=device,
                          hidden=cfg["adapter"]["hidden"], backbone=args.backbone,
                          adapter_ckpt=adapter_ckpt_path(cfg, sc["source"], args.backbone))
    theta0 = model.snapshot()
    alphas = [float(a) for a in args.alphas.split(",")]

    rows = []
    for i in range(n):
        x, gt, stem = stream.take(i)
        xb, gtb = x.unsqueeze(0).to(device), gt.unsqueeze(0).to(device)
        model.restore(theta0)

        # --- self-supervised gradient
        model.zero_grad(set_to_none=True)
        loss_ss = self_supervised_loss(model, xb, prior=prior, with_consistency=True)
        loss_ss.backward()
        g_ss = flat_grad(model).clone()
        ss_norm = float(g_ss.norm())

        # --- supervised gradient (label oracle; diagnostic only)
        model.restore(theta0)
        model.zero_grad(set_to_none=True)
        y = model(xb)
        loss_sup = supervised_loss(y, gtb)
        loss_sup.backward()
        g_sup = flat_grad(model).clone()
        sup_norm = float(g_sup.norm())

        cos = float(torch.nn.functional.cosine_similarity(g_ss, g_sup, dim=0)) \
            if ss_norm > 1e-12 and sup_norm > 1e-12 else float("nan")

        with torch.no_grad():
            m0 = M.per_image(model(xb)[0].cpu(), gt)

        # --- line search along the self-supervised step
        curve = {}
        for a in alphas:
            apply_flat(model, theta0, g_ss, a)
            with torch.no_grad():
                m = M.per_image(model(xb)[0].cpu(), gt)
            curve[a] = m["psnr"] - m0["psnr"]

        # The best ss step length has to be known before the supervised
        # reference step can be taken at the same length -- computing it after
        # the reference step (as an earlier version did) is an UnboundLocalError.
        best_a = max(curve.items(), key=lambda kv: kv[1])

        # --- supervised descent at the SAME step length as the best ss step, so
        # the two are directly comparable (a full-norm step would explode)
        apply_flat(model, theta0, g_sup, -best_a[0])
        with torch.no_grad():
            m_sup = M.per_image(model(xb)[0].cpu(), gt)
        sup_step = m_sup["psnr"] - m0["psnr"]

        model.restore(theta0)
        rows.append({
            "stem": stem, "cos": cos, "g_ss_norm": ss_norm, "g_sup_norm": sup_norm,
            "loss_ss": float(loss_ss), "loss_sup": float(loss_sup),
            "psnr_frozen": m0["psnr"], "curve": curve,
            "best_alpha": best_a[0], "best_gain": best_a[1],
            "gain_at_1.0": curve.get(1.0), "sup_step_gain": sup_step,
        })
        print(f"  {stem[:32]:<32} cos={cos:+.3f} |g_ss|={ss_norm:.2e} |g_sup|={sup_norm:.2e} "
              f"best_a={best_a[0]:+.2f} ({best_a[1]:+.3f} dB) sup_step={sup_step:+.3f} dB", flush=True)

    cos_arr = np.array([r["cos"] for r in rows], dtype=np.float64)
    best = np.array([r["best_gain"] for r in rows], dtype=np.float64)
    at1 = np.array([r["gain_at_1.0"] for r in rows], dtype=np.float64)
    step = np.array([r["sup_step_gain"] for r in rows], dtype=np.float64)
    res = {
        "stream": args.stream, "dev": f"{probe_ds}/dev", "n": n, "backbone": args.backbone,
        "cos_mean": float(np.nanmean(cos_arr)), "cos_median": float(np.nanmedian(cos_arr)),
        "cos_positive_frac": float(np.nanmean(cos_arr > 0)),
        "gain_at_alpha_1_mean": float(at1.mean()),
        "best_gain_mean": float(best.mean()),
        "best_alpha_median": float(np.median([r["best_alpha"] for r in rows])),
        "sup_step_gain_mean": float(step.mean()),
        "verdict": {
            "direction_aligned": bool(np.nanmean(cos_arr) > 0.10),
            "line_search_recovers_gain": bool(best.mean() > 0.15),
            "diagnosis": ("scale_only: rescale the ss step" if (np.nanmean(cos_arr) > 0.10
                          and best.mean() > 0.15)
                          else "objective_misaligned: replace/redesign the loss"),
        },
        "rows": rows,
    }
    out = os.path.join(run_dir(cfg, "align", f"{args.stream}{backbone_suffix(args.backbone)}"),
                       "grad_align.json")
    save_json(res, out)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
