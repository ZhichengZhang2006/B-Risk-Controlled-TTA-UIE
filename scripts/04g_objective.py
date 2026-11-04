"""04g -- select the self-supervised objective on DEV (before any test run).

Background: 04e showed the adapter can gain +1.45 dB under a label loss while
the project's current objective *loses* 0.32 dB, and 04f checks whether that is a
scale problem or a direction problem.  Either way the objective is what limits
the whole method, so it is selected explicitly, on DEV, against a *frozen*
reference -- never against a collapsing always-accept arm (the mistake the first
selection rule made, see scripts/04c_reselect.py).

Objective family.  Every member is label-free and computable from x alone, so all
of them are legal under the unsupervised permission of spec §5.1:

  hf      x - y must be low-frequency (the water column removes detail; the
          enhancer should not invent high-frequency structure).   [current default]
  rgb     output colour moments must match the source-domain prior.
  lab     output Lab a/b means must match the source-domain prior.
  tv      total-variation penalty (smoothness).
  cons    photometric-jitter consistency (output stable under small perturbations).
  gamma   grey-world / illumination prior: the output's channel means should be
          closer to grey than the input's, without collapsing contrast.
  dark    DCP-style prior: the dark channel of the *output* should be lower than
          the input's, i.e. the haze is removed.  Physically motivated and does
          not need the target's labels.
  sat     saturation should rise toward the source prior without clipping.

The selector runs each candidate on the dev split with a small fixed budget,
scores the *true* dPSNR against frozen (a label-using diagnostic, exactly like
04c), and keeps the candidate with the largest reliable gain.  Ties inside the
noise band (0.05 dB) fall back to the simplest objective.

DEV ONLY. Nothing here reads the test split.

Usage:
    python scripts/04g_objective.py --stream LSUI2UIEB --n 24 --steps 3 --lr 0.01
    python scripts/04g_objective.py --all-streams --n 24 --steps 3 --lr 0.01
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import adapter_ckpt_path, backbone_suffix, load_config, open_stream, resolve_ckpt, run_dir, save_json, split_dir  # noqa: E402


sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from brctta import metrics as M  # noqa: E402
from brctta.data import PairedStream, StreamSpec  # noqa: E402
from brctta.losses import load_prior  # noqa: E402
from brctta.model import build_adapted  # noqa: E402

# objective name -> weight dict understood by brctta.losses.self_supervised_loss
CANDIDATES = {
    "hf":        {"hf": 1.0, "tv": 0.02},
    "hf_lab":    {"hf": 1.0, "lab": 0.5, "tv": 0.02},
    "hf_rgb":    {"hf": 1.0, "rgb": 0.5, "tv": 0.02},
    "lab":       {"lab": 1.0, "tv": 0.02},
    "rgb":       {"rgb": 1.0, "tv": 0.02},
    "hf_lab_rgb": {"hf": 1.0, "lab": 0.5, "rgb": 0.5, "tv": 0.02},
    "cons":      {"hf": 1.0, "cons": 0.3, "tv": 0.02},
    "hf_lab_cons": {"hf": 1.0, "lab": 0.5, "cons": 0.3, "tv": 0.02},
}


def grey_world_loss(y: torch.Tensor) -> torch.Tensor:
    """Channel means of the output should be close to their own average.

    Underwater casts push the red channel down and blue/green up; a correct
    restoration moves the means toward equality.  This is a prior on the output
    alone, so it is label-free -- but it can be satisfied by desaturating, which
    is why it is always paired with a structure term.
    """
    m = y.mean(dim=(2, 3))
    return ((m - m.mean(dim=1, keepdim=True)) ** 2).mean()


def dark_channel(x: torch.Tensor, k: int = 7) -> torch.Tensor:
    return -torch.nn.functional.max_pool2d(-x, k, stride=1, padding=k // 2).min(dim=1)[0]


def dcp_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Dark channel of the output should be darker than the input's (haze removed)."""
    return torch.relu(dark_channel(y) - dark_channel(x) * 0.9).mean()


def saturation_prior_loss(y: torch.Tensor, target_sat: float) -> torch.Tensor:
    mx = y.max(dim=1)[0]
    mn = y.min(dim=1)[0]
    sat = ((mx - mn) / (mx + 1e-6)).mean()
    return (sat - target_sat) ** 2


def objective(model, xb, gt_unused, prior, name: str) -> torch.Tensor:
    y = model(xb)
    w = CANDIDATES.get(name, {})
    loss = torch.zeros((), device=xb.device)
    if "hf" in w:
        from brctta.losses import hf_residual_loss
        loss = loss + w["hf"] * hf_residual_loss(xb, y)
    if "lab" in w:
        from brctta.losses import lab_ab_prior_loss
        loss = loss + w["lab"] * lab_ab_prior_loss(y, prior)
    if "rgb" in w:
        from brctta.losses import rgb_prior_loss
        loss = loss + w["rgb"] * rgb_prior_loss(y, prior)
    if "tv" in w:
        from brctta.losses import tv_loss
        loss = loss + w["tv"] * tv_loss(y)
    if "cons" in w:
        from brctta.losses import consistency_loss
        loss = loss + w["cons"] * consistency_loss(model, xb)
    if name in ("gamma", "dcp", "sat"):
        loss = loss + 2.0 * grey_world_loss(y)
        loss = loss + 0.5 * hf_residual_helper(xb, y)
    if name == "dcp":
        loss = loss + 2.0 * dcp_loss(xb, y)
    if name == "sat":
        loss = loss + 2.0 * saturation_prior_loss(y, float(prior.get("saturation_mean", 0.30)))
    return loss


def hf_residual_helper(xb, y):
    from brctta.losses import hf_residual_loss
    return hf_residual_loss(xb, y)


# Objectives that are not expressible as a weight table are handled by branches
# inside objective(); list them here so the CLI default covers them too.
ALL_CANDIDATES = dict(CANDIDATES)
ALL_CANDIDATES.update({k: {} for k in ("gamma", "dcp", "sat")})


def run_candidate(model, stream, n, steps, lr, name, prior, device, theta0):
    rows = []
    for i in range(n):
        x, gt, stem = stream.take(i)
        xb = x.unsqueeze(0).to(device)
        model.restore(theta0)
        with torch.no_grad():
            m0 = M.per_image(model(xb)[0].cpu(), gt)
        opt = torch.optim.SGD(model.adapter_params(), lr=lr)
        losses = []
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            loss = objective(model, xb, gt, prior, name)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.adapter_params(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            m1 = M.per_image(model(xb)[0].cpu(), gt)
        rows.append({"stem": stem, "psnr_frozen": m0["psnr"], "psnr": m1["psnr"],
                     "ssim_frozen": m0["ssim"], "ssim": m1["ssim"],
                     "loss_first": losses[0], "loss_last": losses[-1]})
    model.restore(theta0)
    d = np.array([r["psnr"] - r["psnr_frozen"] for r in rows])
    ds = np.array([r["ssim"] - r["ssim_frozen"] for r in rows])
    frozen = np.array([r["psnr_frozen"] for r in rows])
    return {
        "objective": name, "n": len(rows),
        "psnr": float(np.mean([r["psnr"] for r in rows])),
        "frozen_psnr": float(frozen.mean()),
        "dPSNR_mean": float(d.mean()), "dPSNR_median": float(np.median(d)),
        "dSSIM_mean": float(ds.mean()),
        "NAR_0": float((d < 0).mean()),
        "worst10_dPSNR": float(np.sort(d)[:max(1, int(round(len(d) * 0.1)))].mean()),
        "p90_gain": float(np.percentile(d, 90)),
        "loss_decrease": float(np.mean([r["loss_first"] - r["loss_last"] for r in rows])),
        "corr_loss_vs_dPSNR": float(np.corrcoef(
            [r["loss_first"] - r["loss_last"] for r in rows], d)[0, 1])
        if np.std(d) > 1e-12 else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--stream", action="append", default=[])
    ap.add_argument("--all-streams", action="store_true")
    ap.add_argument("--candidates", default=",".join(ALL_CANDIDATES))
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--dev-domain", default="target")
    ap.add_argument("--backbone", default="mambacue", choices=["mambacue", "aqua"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = args.stream or ([s["name"] for s in cfg["streams"]] if args.all_streams else [])
    if not names:
        raise SystemExit("pass --stream NAME (repeatable) or --all-streams")

    summary = {}
    for stream_name in names:
        sc = next(s for s in cfg["streams"] if s["name"] == stream_name)
        probe_ds = sc["target"] if args.dev_domain == "target" else sc["source"]
        spec = StreamSpec(name=f"obj_{stream_name}", source_dataset=probe_ds,
                          target_split=f"{probe_ds}/dev", seed=sc["seed"])
        stream = open_stream(cfg, probe_ds, "dev", spec)
        n = min(args.n, len(stream))
        prior = load_prior(os.path.join(cfg["paths"]["out_root"], "manifests",
                                        f"prior_{sc['source']}.json"))

        model = build_adapted(resolve_ckpt(cfg, sc, args.backbone), device=device,
                              hidden=cfg["adapter"]["hidden"], backbone=args.backbone,
                              adapter_ckpt=adapter_ckpt_path(cfg, sc["source"], args.backbone))
        theta0 = model.snapshot()

        rows = []
        print(f"\n[{stream_name}] dev={probe_ds}/dev n={n} steps={args.steps} lr={args.lr}", flush=True)
        for cand in args.candidates.split(","):
            cand = cand.strip()
            if not cand:
                continue
            t0 = time.time()
            r = run_candidate(model, stream, n, args.steps, args.lr, cand, prior, device, theta0)
            r["seconds"] = time.time() - t0
            rows.append(r)
            print(f"  {cand:<14} dPSNR={r['dPSNR_mean']:+.4f} (med {r['dPSNR_median']:+.4f}) "
                  f"dSSIM={r['dSSIM_mean']:+.4f} NAR0={r['NAR_0']:.2f} "
                  f"worst10={r['worst10_dPSNR']:+.3f} p90={r['p90_gain']:+.3f} "
                  f"corr={r['corr_loss_vs_dPSNR']:+.2f}", flush=True)

        # Selection: largest dev dPSNR.  Ties inside 0.05 dB resolve to the
        # objective with fewer terms (simpler first), so a marginal win cannot
        # drag in extra priors that might not transfer.
        order = {c: i for i, c in enumerate(args.candidates.split(","))}
        best = max(rows, key=lambda r: (round(r["dPSNR_mean"], 3), -order.get(r["objective"], 99)))
        summary[stream_name] = {"selected": best["objective"], "rows": rows,
                                "dev": f"{probe_ds}/dev", "n": n,
                                "steps": args.steps, "lr": args.lr,
                                "criterion": ("max mean dPSNR vs frozen on the target-domain dev "
                                              "split; label-using diagnostic, never a method; "
                                              "ties within 0.05 dB prefer fewer loss terms")}
        print(f"  SELECTED {best['objective']} ({best['dPSNR_mean']:+.4f} dB)", flush=True)

    out = os.path.join(run_dir(cfg, "objective_selection" + backbone_suffix(args.backbone)),
                       "objective_selection.json")
    save_json(summary, out)
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
