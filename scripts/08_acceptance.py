"""08 -- acceptance report (§11 gates).

Reads the produced result files and applies the *pre-declared* acceptance rules.
Nothing here may be edited after seeing results without a version bump (the file
hash is recorded in the report itself).

Gates (§11)
  G1..G6 : for the three locked transfers, PSNR and SSIM must trail the
           comparable SOTA reference by at most 5% relative (g_m <= 0.05).
  G7     : no substitution of an MSE-based criterion; MSE is reported per image
           but the gate is evaluated on PSNR/SSIM only.
  G8     : every stream must have a valid replay check.
  G9     : the action log must show the controller is not degenerate
           (accept_rate in (0, 1)) on at least two of three streams.
  G10    : on the primary stream, our gate must beat always_accept on worst-10%
           dPSNR and reduce NAR_0 -- the core claim, stated before the fact.

Usage:
    python scripts/08_acceptance.py [--config configs/main.yaml]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_config, run_dir, save_json  # noqa: E402



def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def rel_gap(ours: float, ref: float) -> float:
    """Relative shortfall in the metric's own units (0 = equal, 0.05 = 5% worse)."""
    if ref is None or ref <= 0:
        return float("nan")
    return (ref - ours) / ref


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    # comparable SOTA reference per (stream, metric).  These are the strongest
    # *same-protocol* numbers from docs/sota_registry.csv; where no same-protocol
    # number exists the gate is marked not_applicable rather than fudged.
    sota = read_json(os.path.join(ROOT, "configs", "sota_reference.json")) or {}
    tol = float((sota.get("_meta") or {}).get("tolerance", 0.05))

    gates = []
    streams = {}
    for st in cfg["streams"]:
        name = st["name"]
        d = run_dir(cfg, name)
        res = read_json(os.path.join(d, "online_results.json"))
        rep = read_json(os.path.join(d, "replay_check.json"))
        streams[name] = {"online": bool(res), "replay": rep}

        if not res:
            gates.append({"id": f"G_{name}_present", "ok": False, "detail": "no online_results.json"})
            continue

        methods = {m["method"]: m for m in res["methods"]}
        ours = methods.get("ours")
        always = methods.get("always_accept")
        frozen = methods.get("source_frozen")

        ref = (sota.get(name) or {})
        for metric in ("psnr", "ssim"):
            ref_v = ref.get(metric)
            ours_v = ours.get(f"{metric}_prequential") if ours else None
            if ref_v is None or ours_v is None:
                gates.append({
                    "id": f"G_{name}_{metric}", "ok": None,
                    "detail": "no same-protocol SOTA reference pinned" if ref_v is None else "missing ours",
                })
                continue
            g = rel_gap(ours_v, ref_v)
            gates.append({
                "id": f"G_{name}_{metric}", "ok": bool(g <= tol),
                "ours": ours_v, "ref": ref_v, "rel_gap": g, "tolerance": tol,
                "abs_gap": ours_v - ref_v,
            })

        if rep:
            gates.append({"id": f"G_{name}_replay", "ok": bool(rep.get("ok")), "detail": rep})
        else:
            gates.append({"id": f"G_{name}_replay", "ok": False, "detail": "no replay_check.json"})

        if ours and always:
            better_worst = ours["worst10_dPSNR"] > always["worst10_dPSNR"]
            lower_nar = ours["NAR_0"] < always["NAR_0"]
            gates.append({
                "id": f"G_{name}_gate_beats_always",
                "ok": bool(better_worst and lower_nar),
                "ours_worst10": ours["worst10_dPSNR"], "always_worst10": always["worst10_dPSNR"],
                "ours_NAR0": ours["NAR_0"], "always_NAR0": always["NAR_0"],
            })
            gates.append({
                "id": f"G_{name}_nondegenerate",
                "ok": bool(0.0 < ours["accept_rate"] < 1.0),
                "accept_rate": ours["accept_rate"],
            })
            if frozen:
                gates.append({
                    "id": f"G_{name}_not_worse_than_frozen",
                    "ok": bool(ours["psnr_prequential"] >= frozen["psnr_prequential"] - 0.05),
                    "ours": ours["psnr_prequential"], "frozen": frozen["psnr_prequential"],
                })

    def _ok(g):
        return g["ok"] is not False

    passed = [g for g in gates if g["ok"] is True]
    failed = [g for g in gates if g["ok"] is False]
    skipped = [g for g in gates if g["ok"] is None]

    report = {
        "tolerance": tol,
        "n_gates": len(gates),
        "n_passed": len(passed),
        "n_failed": len(failed),
        "n_not_applicable": len(skipped),
        "verdict": "pass" if not failed and passed else ("fail" if failed else "incomplete"),
        "gates": gates,
        "streams": {k: {"online": v["online"], "replay_ok": bool(v["replay"] and v["replay"].get("ok"))}
                    for k, v in streams.items()},
        "frozen_script_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
        "note": ("MSE is reported per image but never used as an acceptance criterion "
                 "(spec §11). Not-applicable gates mean no pinned same-protocol SOTA "
                 "number exists; they are reported as such, not silently passed."),
    }
    out = os.path.join(cfg["paths"]["out_root"], "acceptance_report.json")
    save_json(report, out)

    print(f"{'gate':<34}{'result':<12}detail")
    print("-" * 90)
    for g in gates:
        r = "PASS" if g["ok"] is True else ("FAIL" if g["ok"] is False else "N/A")
        detail = {k: v for k, v in g.items() if k not in ("id", "ok", "detail")}
        if "detail" in g and not detail:
            detail = g["detail"]
        print(f"{g['id']:<34}{r:<12}{detail}")
    print("-" * 90)
    print(f"verdict: {report['verdict']}  ({len(passed)} pass / {len(failed)} fail / {len(skipped)} n/a)")
    print("wrote", out)
    return 0 if report["verdict"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
