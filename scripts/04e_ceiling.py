"""04e -- diagnose WHY the gateable headroom is ~0 dB.

Observed on dev (all three streams): the image-wise ceiling of the current
proposal is only +0.006..+0.049 dB above simply freezing the model.  Either

  (a) the adapter has no capacity to improve these images at all, or
  (b) the adapter can improve them, but our self-supervised objective points the
      wrong way (spec §4 warns exactly about this: a shrinking unlabeled loss is
      not evidence of a real gain).

This script separates the two by measuring three ceilings on the same DEV images,
each with the same adapter, budget and rollback machinery:

  supervised_oracle   k steps of *label* loss (L1+SSIM) on each image, reset each
                      image.  Needs the ground truth, so it is NOT a method -- it
                      is the plasticity ceiling: what this adapter could do if the
                      loss were perfect.
  unsupervised        k steps of the project's self-supervised objective (what the
                      real system uses), reset each image.
  frozen              no update.

Reported per stream: mean dPSNR of each arm against frozen, and the correlation
between the self-supervised loss decrease and the true PSNR change.  A high
supervised ceiling with a near-zero unsupervised one means the objective is the
bottleneck, not the adapter.

DEV ONLY. The test split is never read here.

Usage:
    python scripts/04e_ceiling.py --stream LSUI2UIEB --n 20 --steps 3 --lr 0.01
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
from brctta.losses import load_prior, self_supervised_loss  # noqa: E402
from brctta.model import build_adapted  # noqa: E402


def supervised_loss(y: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """L1 + (1 - SSIM-ish) surrogate: the loss a fully supervised step would use."""
    l1 = (y - gt).abs().mean()
    # cheap differentiable structural term: 1 - normalised cross correlation
    a = y - y.mean(dim=(2, 3), keepdim=True)
    b = gt - gt.mean(dim=(2, 3), keepdim=True)
    ncc = (a * b).sum(dim=(2, 3)) / (a.norm(dim=(2, 3)) * b.norm(dim=(2, 3)) + 1e-6)
    return l1 + 0.2 * (1.0 - ncc.mean())


def run_arm(model, stream, n, steps, lr, arm, prior, device):
    """Episodic: reset to theta_0 before every image, so each image sees k fresh steps."""
    theta0 = model.snapshot()
    out = []
    for i in range(n):
        x, gt, stem = stream.take(i)
        xb = x.unsqueeze(0).to(device)
        gtb = gt.unsqueeze(0).to(device)
        model.restore(theta0)
        with torch.no_grad():
            y0 = model(xb)
        m0 = M.per_image(y0[0].cpu(), gt)

        if arm == "frozen" or steps <= 0:
            out.append({"stem": stem, "psnr": m0["psnr"], "ssim": m0["ssim"],
                        "psnr_frozen": m0["psnr"], "loss_first": None, "loss_last": None,
                        "gnorm": 0.0})
            continue

        opt = torch.optim.SGD(model.adapter_params(), lr=lr, momentum=0.0)
        losses, gnorm = [], 0.0
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            y = model(xb)
            loss = supervised_loss(y, gtb) if arm == "supervised" else \
                self_supervised_loss(model, xb, prior=prior, with_consistency=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.adapter_params(), 1.0))
            opt.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            y1 = model(xb)
        m1 = M.per_image(y1[0].cpu(), gt)
        out.append({"stem": stem, "psnr": m1["psnr"], "ssim": m1["ssim"],
                    "psnr_frozen": m0["psnr"], "loss_first": losses[0],
                    "loss_last": losses[-1], "gnorm": gnorm})
        model.restore(theta0)
    return out


def summarize(rows, arm):
    p = np.array([r["psnr"] for r in rows], dtype=np.float64)
    f = np.array([r["psnr_frozen"] for r in rows], dtype=np.float64)
    d = p - f
    s = np.array([r["ssim"] for r in rows], dtype=np.float64)
    res = {
        "arm": arm, "n": len(rows),
        "psnr": float(p.mean()), "psnr_frozen": float(f.mean()),
        "dPSNR_mean": float(d.mean()), "dPSNR_median": float(np.median(d)),
        "NAR_0": float((d < 0).mean()),
        "worst10_dPSNR": float(np.sort(d)[:max(1, int(round(len(d) * 0.1)))].mean()),
        "delta_p90": float(np.percentile(d, 90)),
    }
    if rows[0]["loss_first"] is not None:
        lf = np.array([r["loss_first"] for r in rows], dtype=np.float64)
        ll = np.array([r["loss_last"] for r in rows], dtype=np.float64)
        res["loss_decrease_mean"] = float((lf - ll).mean())
        # does "the unlabeled loss went down" predict "the image got better"?
        if np.std(ll - lf) > 1e-12 and np.std(d) > 1e-12:
            res["corr_lossdecr_vs_dPSNR"] = float(np.corrcoef(lf - ll, d)[0, 1])
        else:
            res["corr_lossdecr_vs_dPSNR"] = float("nan")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--stream", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--dev-domain", default="target")
    ap.add_argument("--backbone", default="mambacue", choices=["mambacue", "aqua"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    sc = next(s for s in cfg["streams"] if s["name"] == args.stream)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    probe_ds = sc["target"] if args.dev_domain == "target" else sc["source"]
    split = "dev"

    spec = StreamSpec(name=f"ceiling_{args.stream}", source_dataset=probe_ds,
                      target_split=f"{probe_ds}/{split}", seed=sc["seed"])
    stream = open_stream(cfg, probe_ds, split, spec)
    n = min(args.n, len(stream))
    prior = load_prior(os.path.join(cfg["paths"]["out_root"], "manifests", f"prior_{sc['source']}.json"))

    ckpt = resolve_ckpt(cfg, sc, args.backbone)
    model = build_adapted(ckpt, device=device, hidden=cfg["adapter"]["hidden"], backbone=args.backbone,
                          adapter_ckpt=adapter_ckpt_path(cfg, sc["source"], args.backbone))

    print(f"[{args.stream}] dev={probe_ds}/{split} n={n} steps={args.steps} lr={args.lr} "
          f"backbone={args.backbone}", flush=True)
    results = {"stream": args.stream, "dev": f"{probe_ds}/{split}", "n": n,
               "steps": args.steps, "lr": args.lr, "backbone": args.backbone, "arms": {}}
    for arm in ("frozen", "supervised", "unsupervised"):
        t0 = time.time()
        rows = run_arm(model, stream, n, args.steps, args.lr, arm, prior, device)
        s = summarize(rows, arm)
        s["seconds"] = time.time() - t0
        results["arms"][arm] = s
        print(f"  {arm:<13} psnr={s['psnr']:.4f} dPSNR={s['dPSNR_mean']:+.4f} "
              f"median={s['dPSNR_median']:+.4f} NAR0={s['NAR_0']:.2f} "
              f"worst10={s['worst10_dPSNR']:+.3f} p90={s['delta_p90']:+.3f}"
              + (f" corr(loss,psnr)={s['corr_lossdecr_vs_dPSNR']:+.3f}"
                 if "corr_lossdecr_vs_dPSNR" in s else ""), flush=True)

    sup = results["arms"]["supervised"]["dPSNR_mean"]
    uns = results["arms"]["unsupervised"]["dPSNR_mean"]
    results["verdict"] = {
        "supervised_ceiling_dB": sup,
        "unsupervised_dB": uns,
        "adapter_can_improve": bool(sup > 0.10),
        "objective_is_bottleneck": bool(sup > 0.10 and uns < 0.25 * sup),
        "note": ("supervised uses the ground truth, so it is a plasticity ceiling only; "
                 "it is never a method and never enters the comparison set."),
    }
    out = os.path.join(run_dir(cfg, "ceiling", f"{args.stream}{backbone_suffix(args.backbone)}"),
                       f"ceiling_{args.steps}x{args.lr}.json")
    save_json(results, out)
    print(json.dumps(results["verdict"], indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
