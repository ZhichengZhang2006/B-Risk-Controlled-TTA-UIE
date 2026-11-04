"""Build `experiment_ledger.csv` from everything in `runs/`.

Spec §13 requires an auditable ledger: every experiment, what it was for, the
exact configuration, and where its numbers came from.  Rather than hand-maintain
a spreadsheet (which drifts from the code), the ledger is *derived* from the
result files each script already writes.  If a number is not in a result file it
does not appear here -- that is the point.

Columns
-------
run_id, kind, stream, backbone, dev_or_test, n, steps, lr, primary_metric,
primary_value, secondary_metric, secondary_value, seconds, path, note

`dev_or_test` is the honest split label.  Anything measured on `test` is a
protocol number; anything on `dev` is a selection number and must never be
quoted as a result.
"""

import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from common import load_config  # noqa: E402

COLUMNS = ["run_id", "kind", "stream", "backbone", "dev_or_test", "n", "steps", "lr",
           "primary_metric", "primary_value", "secondary_metric", "secondary_value",
           "seconds", "path", "note"]


def _load(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _row(**kw):
    return {c: kw.get(c, "") for c in COLUMNS}


def collect(cfg):
    out_root = cfg["paths"]["out_root"]
    rows = []

    # --- online evaluation results (the main table) -------------------------
    for path in sorted(glob.glob(os.path.join(out_root, "online_*", "online_results.json"))):
        d = _load(path)
        if not d:
            continue
        run = os.path.basename(os.path.dirname(path))
        split = "test" if "test" in str(d.get("split", "test")) else "dev"
        for m in d.get("methods", []):
            rows.append(_row(
                run_id=f"{run}:{m.get('method')}", kind="online_eval",
                stream=d.get("stream", ""), backbone=d.get("backbone", "mambucue"),
                dev_or_test=split, n=m.get("n", ""),
                steps=d.get("adapt", {}).get("steps", ""), lr=d.get("adapt", {}).get("lr", ""),
                primary_metric="psnr", primary_value=round(float(m.get("psnr", 0)), 4),
                secondary_metric="ssim", secondary_value=round(float(m.get("ssim", 0)), 5),
                seconds=round(float(m.get("seconds", 0) or 0), 1),
                path=os.path.relpath(path, out_root),
                note=f"theta0={d.get('theta0', '')}; accept={m.get('accept_rate', '')}; "
                     f"reset={m.get('reset_rate', '')}"))

    # --- adaptation selection (dev) ----------------------------------------
    for path in sorted(glob.glob(os.path.join(out_root, "selection_*", "adapt_selection.json"))):
        d = _load(path)
        if not d:
            continue
        run = os.path.basename(os.path.dirname(path))
        for r in d.get("rows", []):
            rows.append(_row(
                run_id=f"{run}:{r.get('cell')}", kind="selection_cell",
                stream=d.get("stream", ""), backbone=d.get("backbone", "mambucue"),
                dev_or_test="dev", n=d.get("n", ""),
                steps=r.get("steps", ""), lr=r.get("lr", ""),
                primary_metric="bound_vs_frozen_dB",
                primary_value=round(float(r.get("bound_vs_frozen_dB", 0) or 0), 4),
                secondary_metric="always_dPSNR",
                secondary_value=round(float(r.get("always_dPSNR", 0) or 0), 4),
                path=os.path.relpath(path, out_root),
                note=("SELECTED" if r.get("cell") == d.get("selected_cell") else "")))

    # --- ceiling / alignment / objective diagnosis (dev) -------------------
    for pat, kind in (("ceiling_*", "ceiling"), ("align_*", "grad_align"),
                      ("objective_selection*", "objective")):
        for path in sorted(glob.glob(os.path.join(out_root, pat, "*.json"))):
            d = _load(path)
            if not d:
                continue
            run = os.path.basename(os.path.dirname(path))
            arms = d.get("arms") or d.get("candidates") or {}
            if isinstance(arms, dict):
                for name, a in arms.items():
                    if not isinstance(a, dict):
                        continue
                    rows.append(_row(
                        run_id=f"{run}:{name}", kind=kind, stream=d.get("stream", ""),
                        backbone=d.get("backbone", "mambucue"), dev_or_test="dev",
                        n=a.get("n", d.get("n", "")),
                        primary_metric="dPSNR",
                        primary_value=round(float(a.get("dPSNR", a.get("d_psnr", 0)) or 0), 4),
                        secondary_metric="NAR0",
                        secondary_value=round(float(a.get("nar0", a.get("NAR_0", 0)) or 0), 4),
                        path=os.path.relpath(path, out_root),
                        note=str(a.get("verdict", ""))[:120]))

    # --- stats-match / affine redegrade (dev) ------------------------------
    for path in sorted(glob.glob(os.path.join(out_root, "affine_redegrade*", "affine_redegrade.json"))):
        d = _load(path)
        if not d:
            continue
        run = os.path.basename(os.path.dirname(path))
        for name, a in (d.get("arms") or {}).items():
            rows.append(_row(
                run_id=f"{run}:{name}", kind="affine_redegrade", stream=d.get("stream", ""),
                backbone=d.get("backbone", "mambucue"), dev_or_test="dev", n=d.get("n", ""),
                primary_metric="dPSNR", primary_value=round(float(a.get("dPSNR", 0)), 4),
                secondary_metric="frac_pos", secondary_value=round(float(a.get("frac_pos", 0)), 4),
                path=os.path.relpath(path, out_root), note=""))

    # --- strong frozen cross-domain comparators (test) ---------------------
    for path in sorted(glob.glob(os.path.join(out_root, "strong_frozen_*", "*.json"))):
        d = _load(path)
        if not d:
            continue
        rows.append(_row(
            run_id=os.path.basename(path).replace(".json", ""), kind="strong_frozen",
            stream=f"{d.get('ckpt_dataset')}2{d.get('eval_dataset')}",
            backbone=d.get("backbone", "aqua"), dev_or_test="test", n=d.get("n", ""),
            primary_metric="psnr", primary_value=round(float(d.get("psnr", 0)), 4),
            secondary_metric="ssim", secondary_value=round(float(d.get("ssim", 0)), 5),
            seconds=round(float(d.get("seconds", 0) or 0), 1),
            path=os.path.relpath(path, out_root), note=f"role={d.get('role', '')}"))

    # --- theta0 / pretrain -------------------------------------------------
    for path in sorted(glob.glob(os.path.join(out_root, "theta0_eval_*", "theta0_eval.json"))):
        d = _load(path)
        if not d:
            continue
        run = os.path.basename(os.path.dirname(path))
        # `splits` is a LIST of {split, frozen{...}, adapter{...}, delta_psnr, delta_ssim}
        for a in (d.get("splits") or []):
            name = a.get("split", "")
            rows.append(_row(
                run_id=f"{run}:{name}", kind="theta0", stream=name,
                backbone=d.get("backbone", "mambucue"),
                dev_or_test=("test" if name.endswith("/test") else "dev"),
                n=(a.get("frozen") or {}).get("n", ""),
                primary_metric="dPSNR", primary_value=round(float(a.get("delta_psnr", 0)), 4),
                secondary_metric="dSSIM", secondary_value=round(float(a.get("delta_ssim", 0)), 5),
                path=os.path.relpath(path, out_root),
                note=f"frozen {(a.get('frozen') or {}).get('psnr', 0):.4f} -> "
                     f"adapter {(a.get('adapter') or {}).get('psnr', 0):.4f}"))

    for path in sorted(glob.glob(os.path.join(out_root, "adapter_pretrain_*", "adapter_pretrain.json"))):
        d = _load(path)
        if not d:
            continue
        rows.append(_row(
            run_id=os.path.basename(os.path.dirname(path)), kind="adapter_pretrain",
            stream=d.get("source", ""), backbone=d.get("backbone", "mambucue"),
            dev_or_test="source_train", n=d.get("pairs", ""), steps=d.get("epochs", ""),
            lr=d.get("lr", ""), primary_metric="final_loss",
            primary_value=round(float(d.get("final_loss", 0) or 0), 6),
            secondary_metric="sha256", secondary_value=str(d.get("sha256", ""))[:16],
            path=os.path.relpath(path, out_root), note=""))

    # --- degrader ---------------------------------------------------------
    for path in sorted(glob.glob(os.path.join(out_root, "degrader_*", "degrader.json"))):
        d = _load(path)
        if not d:
            continue
        rows.append(_row(
            run_id=os.path.basename(os.path.dirname(path)), kind="degrader",
            stream=d.get("source", ""), dev_or_test="source_train", n=d.get("pairs", ""),
            steps=d.get("epochs", ""), primary_metric="final_loss",
            primary_value=round(float(d.get("final_loss", 0) or 0), 6),
            secondary_metric="sha256", secondary_value=str(d.get("sha256", ""))[:16],
            path=os.path.relpath(path, out_root), note="clean->degraded residual model"))

    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    cfg = load_config()
    rows = collect(cfg)
    out = args.out or os.path.join(cfg["paths"]["out_root"], "experiment_ledger.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"[ledger] {len(rows)} rows -> {out}")
    for k, v in sorted(kinds.items()):
        print(f"   {k:20s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
