"""04c -- re-apply the (fixed) selection rule to already-measured dev cells.

Why this exists: the first selection rule compared the gateable ceiling against
the *always accept* arm.  On UIEB2LSUI that picked 5x0.01, a cell where
always-accept had already collapsed aggregate PSNR from 22.5 dB to 16.9 dB -- the
rule rewarded destruction because a collapsing reference makes any surviving
image look like a gain.

The measurements themselves are fine; only the comparison point was wrong.  So
instead of re-running the (expensive) adaptation cells, this script recomputes
the decision from the logs already on disk plus one cheap extra pass: the FROZEN
score on the same dev stream, i.e. the score of never updating at all.

    gain_vs_frozen = mean_i max(psnr_pre_i, psnr_post_i) - PSNR(frozen)

The old decision is preserved in the output (``rejected_selection``) so the
iteration is auditable and cannot be mistaken for a rule that was always used.
Nothing here reads the test split.

Usage:
    python scripts/04c_reselect.py --stream LSUI2UIEB
    python scripts/04c_reselect.py --all
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
from brctta.baselines import FrozenBaseline  # noqa: E402
from brctta.data import PairedStream, StreamSpec  # noqa: E402
from brctta.model import build_adapted  # noqa: E402

MIN_GAIN_VS_FROZEN = 0.05


def imagewise_bound(log_path: str) -> float:
    recs = [json.loads(l) for l in open(log_path, "r", encoding="utf-8") if l.strip()]
    pre = np.array([r["psnr_pre"] for r in recs], dtype=np.float64)
    post = np.array([r["psnr_post"] for r in recs], dtype=np.float64)
    return float(np.maximum(pre, post).mean())


def frozen_on_dev(cfg, stream_cfg, probe_ds, split, n, device, backbone, log_path):
    spec = StreamSpec(name=f"reselect_{stream_cfg['name']}", source_dataset=probe_ds,
                      target_split=f"{probe_ds}/{split}", seed=stream_cfg["seed"])
    stream = open_stream(cfg, probe_ds, split, spec)
    ckpt = resolve_ckpt(cfg, stream_cfg, backbone)
    model = build_adapted(ckpt, device=device, hidden=cfg["adapter"]["hidden"], backbone=backbone,
                          adapter_ckpt=adapter_ckpt_path(cfg, stream_cfg["source"], backbone))
    eng = FrozenBaseline(model, device=device)
    s = eng.run(stream, limit=n, log_path=log_path)
    return s


def reselect_one(cfg, stream_name: str, device: str, backbone: str) -> dict:
    stream_cfg = next(s for s in cfg["streams"] if s["name"] == stream_name)
    sel_dir = run_dir(cfg, "selection", f"{stream_name}_{stream_cfg['target']}" + backbone_suffix(backbone))
    path = os.path.join(sel_dir, "adapt_selection.json")
    if not os.path.exists(path):
        raise SystemExit(f"no selection file at {path}; run 04b first")
    old = json.load(open(path, encoding="utf-8"))

    probe_ds, split, n = old["dev_domain"], old["split"], old["n"]
    logdir = os.path.join(sel_dir, "logs")
    fz_path = os.path.join(logdir, "frozen.jsonl")
    if not os.path.exists(fz_path):
        fz = frozen_on_dev(cfg, stream_cfg, probe_ds, split, n, device, backbone, fz_path)
    else:
        from brctta.evaluate import summarize

        from brctta.adapt import ActionRecord

        recs = [ActionRecord(**{k: v for k, v in json.loads(l).items()
                                if k in ActionRecord.__dataclass_fields__})
                for l in open(fz_path, encoding="utf-8") if l.strip()]
        fz = summarize(recs, method="source_frozen")
    frozen_psnr = float(fz["psnr_prequential"])

    rows = []
    for r in old["rows"]:
        lp = os.path.join(logdir, f"always_{r['cell']}.jsonl")
        bound = imagewise_bound(lp)
        row = dict(r)
        row["imagewise_bound_psnr"] = bound
        row["imagewise_headroom_dB"] = bound - r["always"]["psnr_prequential"]
        row["frozen_psnr"] = frozen_psnr
        row["bound_vs_frozen_dB"] = bound - frozen_psnr
        row["always_vs_frozen_dB"] = r["always"]["psnr_prequential"] - frozen_psnr
        rows.append(row)

    viable = [r for r in rows if r["bound_vs_frozen_dB"] >= MIN_GAIN_VS_FROZEN]
    if not viable:
        viable = rows
    best = max(viable, key=lambda r: (round(r["bound_vs_frozen_dB"], 4),
                                      -abs(r["always"]["dPSNR_mean"])))

    payload = dict(old)
    payload.update({
        "criterion": ("max gain_vs_frozen = mean(max(psnr_pre,psnr_post)) - PSNR(frozen), "
                      "restricted to cells with gain >= 0.05 dB; dev split only. Rejected "
                      "rules: greedy-oracle headroom (rewards destruction), greedy-oracle "
                      "ceiling (oracle myopia), and image-wise headroom over always-accept "
                      "(rewarded a cell whose always-arm collapsed aggregate PSNR by 5.7 dB)."),
        "min_gain_vs_frozen_dB": MIN_GAIN_VS_FROZEN,
        "frozen": {"psnr_prequential": frozen_psnr,
                   "ssim_prequential": float(fz["ssim_prequential"]),
                   "dPSNR_mean": float(fz.get("dPSNR_mean", 0.0))},
        "rows": rows,
        "selected": best["cell"], "selected_steps": best["steps"], "selected_lr": best["lr"],
        "selected_headroom_dB": best["imagewise_headroom_dB"],
        "selected_gain_vs_frozen_dB": best["bound_vs_frozen_dB"],
        "rejected_selection": {
            "cell": old["selected"], "rule": "max image-wise headroom over always-accept",
            "why_rejected": ("compared against an arm that had already collapsed aggregate "
                             f"PSNR to {min(r['always']['psnr_prequential'] for r in rows):.2f} dB"),
            "bound_vs_frozen_dB": next(r["bound_vs_frozen_dB"] for r in rows
                                       if r["cell"] == old["selected"]),
        },
        "reselect_note": ("Rule fixed after the dev-side measurement, before any test run; "
                          "old decision preserved above. No test data was read."),
    })
    save_json(payload, path)
    print(f"[{stream_name}] frozen {frozen_psnr:.4f} dB")
    for r in rows:
        flag = "  <-- selected" if r["cell"] == best["cell"] else ""
        print(f"  {r['cell']:>10}  bound {r['imagewise_bound_psnr']:.4f}  "
              f"vs frozen {r['bound_vs_frozen_dB']:+.4f} dB  "
              f"(always {r['always_vs_frozen_dB']:+.4f}){flag}")
    print(f"[{stream_name}] SELECTED {best['cell']} steps={best['steps']} lr={best['lr']}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--stream", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--backbone", default="mambacue", choices=["mambacue", "aqua"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = args.stream or ([s["name"] for s in cfg["streams"]] if args.all else [])
    if not names:
        raise SystemExit("pass --stream NAME (repeatable) or --all")
    summary = {}
    for name in names:
        try:
            p = reselect_one(cfg, name, device, args.backbone)
            summary[name] = {"selected": p["selected"], "steps": p["selected_steps"],
                             "lr": p["selected_lr"],
                             "gain_vs_frozen_dB": p["selected_gain_vs_frozen_dB"],
                             "rejected": p["rejected_selection"]["cell"]}
        except SystemExit as exc:
            print(f"[{name}] skipped: {exc}")
    save_json(summary, os.path.join(run_dir(cfg, "selection"), "reselect_summary.json"))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
