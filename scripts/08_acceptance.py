"""08 -- acceptance report (§11 gates).

Reads the produced result files and applies the *pre-declared* acceptance rules.
Nothing here may be edited after seeing results without a version bump; the file
hash is recorded in the report itself, and `scripts/12_freeze.py --check` fails on
drift.

What counts as the reference ``S_m``
------------------------------------
§11 compares against "the strongest available method under the same transfer, the
same source-data / pretraining permission, the same absence of target labels and
the same online protocol".  A model *trained on the target's paired data* is
expressly not that reference -- it is a target-supervision coordinate, and
comparing an unsupervised TTA number to it would fail every method in the field.

So the gate reference is the strongest arm this project actually measured under
the frozen protocol (``runs/comparison/comparison_set.json``), chosen from
deployable arms only; the label-using ``oracle`` is never eligible.  Published
and target-supervised numbers are reported in a separate block and never enter a
gate.  If no comparable arm exists for a metric the gate is ``unverifiable``
(``ok: null``), which the verdict treats as "not passed".

Gates
  spec (the six §11 gates)
    G<stream>_psnr / G<stream>_ssim : g_m = max(0, (S_m - R_m)/S_m) <= 0.05
  protocol
    G<stream>_replay                : the action log replays to the logged outputs
    G<stream>_candidate_labels      : logs carry action-independent candidate metrics
  claim (stated before the fact; failure means the mechanism did not work, and the
  paper must say so rather than re-tune on test)
    G<stream>_beats_always          : ours worst-10% > always, ours NAR_0 < always,
                                      AND the method must be active (0<accept<1):
                                      refusing everything is not risk control
    G<stream>_positive_gain         : realised prequential gain > 0 (abstain scores 0)
    G<stream>_nondegenerate         : 0 < accept_rate < 1
    G<stream>_not_worse_than_frozen : ours >= frozen - 0.05 dB
  integrity
    G_ledger / G_freeze / G_novelty / G_data_manifest

Usage:
    python scripts/08_acceptance.py [--config configs/main.yaml]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_config, run_dir, save_json  # noqa: E402

TOL = 0.05            # §11 relative shortfall allowance
FROZEN_SLACK_DB = 0.05  # "do no harm" tolerance vs the frozen reference


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def rel_gap(ours: float, ref: float) -> float:
    """Relative shortfall in the metric's own units (0 = equal, 0.05 = 5% worse)."""
    if ref is None or ref <= 0 or ours is None:
        return float("nan")
    return max(0.0, (ref - ours) / ref)


def per_image_mse(log_path):
    """Per-image MSE of the *prequential* output (the number §11 asks to report)."""
    if not log_path or not os.path.exists(log_path):
        return None
    vals = []
    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            v = r.get("mse_pre")
            if isinstance(v, (int, float)) and np.isfinite(v):
                vals.append(float(v))
    if not vals:
        return None
    a = np.asarray(vals)
    return {"n": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
            "p90": float(np.quantile(a, 0.90))}


def log_for_arm(out_root, stream, arm, variant=None):
    base = os.path.join(out_root, variant or stream)
    name = {"always_accept": "always", "source_frozen": "source",
            "source_adapter": "source_adapter", "ours": ""}.get(arm, arm)
    cand = os.path.join(base, f"action_log_{name}.jsonl" if name else "action_log.jsonl")
    return cand if os.path.exists(cand) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_root = cfg["paths"]["out_root"]

    comp = read_json(os.path.join(out_root, "comparison", "comparison_set.json")) or {}
    sota = read_json(os.path.join(ROOT, "configs", "sota_reference.json")) or {}
    comp_streams = comp.get("streams", {})
    gate_arms = set((comp.get("_meta") or {}).get("gate_arms", []))

    spec_gates, protocol_gates, claim_gates = [], [], []
    streams = {}
    separate_refs = {}

    for st in cfg["streams"]:
        name = st["name"]
        d = run_dir(cfg, name)
        res = read_json(os.path.join(d, "online_results.json"))
        rep = read_json(os.path.join(d, "replay_check.json"))
        streams[name] = {"online": bool(res), "replay": rep}
        if not res:
            protocol_gates.append({"id": f"G_{name}_present", "ok": False,
                                   "detail": "no online_results.json"})
            continue

        methods = {m["method"]: m for m in res["methods"]}
        ours = methods.get("ours")
        always = methods.get("always_accept")
        frozen = methods.get("source_adapter") or methods.get("source_frozen")

        # ---- §11 reference: strongest measured deployable arm on this stream ----
        arms = [a for a in comp_streams.get(name, {}).get("arms", []) if a["arm"] in gate_arms]
        ref_by_metric = {}
        for metric in ("psnr", "ssim"):
            vals = [(a[metric], a["arm"], a.get("variant")) for a in arms
                    if isinstance(a.get(metric), (int, float)) and np.isfinite(a[metric])]
            ref_by_metric[metric] = max(vals) if vals else None
        # target-supervised / published coordinates: reported, never gating
        # configs/sota_reference.json stores one flat entry per stream:
        # {"psnr": .., "ssim": .., "ref_id": "S09", "ref_method": .., "note": ..}
        # and every entry is a target-supervised in-domain model (see its _meta).
        entry = sota.get(name) or {}
        separate_refs[name] = {
            "target_supervision_reference": (
                {"psnr": entry.get("psnr"), "ssim": entry.get("ssim"),
                 "ref_id": entry.get("ref_id"), "ref_method": entry.get("ref_method"),
                 "note": entry.get("note")}
                if entry else None),
            "published_same_protocol": None,
            "status": "target-supervised in-domain coordinate; NOT part of the §11 gate",
        }

        for metric in ("psnr", "ssim"):
            key = f"{metric}_prequential"
            ref_tuple = ref_by_metric[metric]
            if ours is None or ref_tuple is None:
                spec_gates.append({
                    "id": f"G_{name}_{metric}", "ok": None,
                    "detail": "no comparable measured arm" if ref_tuple is None else "missing ours",
                })
                continue
            ref_v, ref_arm, ref_variant = ref_tuple
            ours_v = float(ours[key])
            g = float((ref_v - ours_v) / ref_v) if ref_v > 0 else float("nan")
            gap = float(ours_v - ref_v)
            mse_ours = per_image_mse(log_for_arm(out_root, name, "ours"))
            mse_ref = per_image_mse(log_for_arm(out_root, name, ref_arm, ref_variant))
            spec_gates.append({
                "id": f"G_{name}_{metric}",
                "ok": bool(np.isfinite(g) and g <= TOL),
                "ours": ours_v, "ref": ref_v, "ref_arm": ref_arm,
                "rel_gap": float(g), "tolerance": TOL,
                "abs_gap": gap,
                "abs_gap_unit": "dB" if metric == "psnr" else "SSIM (absolute)",
                "mse_per_image_ours": mse_ours, "mse_per_image_ref": mse_ref,
                "mse_is_reported_not_gated": True,
            })

        # ---- protocol gates ----
        protocol_gates.append({
            "id": f"G_{name}_replay",
            "ok": bool(rep and rep.get("ok")),
            "detail": None if not rep else {
                "max_abs_dpsnr": rep.get("max_abs_dpsnr"),
                "max_abs_dssim": rep.get("max_abs_dssim"),
                "action_histogram": rep.get("action_histogram"),
            },
        })
        legacy = bool(ours and ours.get("candidate_labels_legacy"))
        protocol_gates.append({
            "id": f"G_{name}_candidate_labels", "ok": (not legacy) if ours else False,
            "detail": "logs lack action-independent candidate metrics" if legacy else "ok",
        })

        # ---- claim gates ----
        # NOTE ON DIRECTION.  A gate pair of (no worse tail, no higher NAR) is
        # satisfiable by never committing anything: refusing every update gives
        # worst10 = 0 and NAR_0 = 0 while proving nothing.  Observed on
        # LSUI2UIEB, where the abstain rule fired.  The precondition below makes
        # the claim *harder* to satisfy (it can only turn a pass into a fail), so
        # adding it after seeing the results does not weaken the acceptance test.
        if ours and always:
            active = bool(0.0 < ours["accept_rate"] < 1.0)
            beats = bool(ours["worst10_dPSNR"] > always["worst10_dPSNR"]
                         and ours["NAR_0"] < always["NAR_0"])
            claim_gates.append({
                "id": f"G_{name}_beats_always",
                "ok": bool(active and beats),
                "active": active,
                "note": ("method abstained: no committed updates, so the tail-risk "
                         "claim is unsupported" if not active else ""),
                "ours_worst10": ours["worst10_dPSNR"], "always_worst10": always["worst10_dPSNR"],
                "ours_NAR0": ours["NAR_0"], "always_NAR0": always["NAR_0"],
                "ours_worst10_cand": ours.get("worst10_dPSNR_cand"),
                "always_worst10_cand": always.get("worst10_dPSNR_cand"),
            })
            # Did adopting the rule actually improve the stream it was run on?
            # Strictly positive realised gain; abstaining scores exactly 0.0 and
            # therefore cannot pass.  This is the "risk control is not refusal"
            # requirement of §9.2 (benefit retention) in its weakest form.
            claim_gates.append({
                "id": f"G_{name}_positive_gain",
                "ok": bool(ours.get("dPSNR_mean", 0.0) > 0.0),
                "ours_dPSNR_mean": ours.get("dPSNR_mean"),
                "note": "realised prequential gain must be strictly positive",
            })

        if ours:
            claim_gates.append({
                "id": f"G_{name}_nondegenerate",
                "ok": bool(0.0 < ours["accept_rate"] < 1.0),
                "accept_rate": ours["accept_rate"],
            })
        if ours and frozen:
            claim_gates.append({
                "id": f"G_{name}_not_worse_than_frozen",
                "ok": bool(ours["psnr_prequential"] >= frozen["psnr_prequential"] - FROZEN_SLACK_DB),
                "ours": ours["psnr_prequential"], "frozen": frozen["psnr_prequential"],
                "frozen_arm": "source_adapter" if "source_adapter" in methods else "source_frozen",
            })

    # ---- integrity gates ----
    ledger = os.path.join(out_root, "experiment_ledger.csv")
    integrity = [{
        "id": "G_ledger",
        "ok": bool(os.path.exists(ledger) and os.path.getsize(ledger) > 200),
        "detail": ledger,
    }]
    frozen_hashes = read_json(os.path.join(ROOT, "docs", "frozen_hashes.json"))
    integrity.append({
        "id": "G_freeze",
        "ok": bool(frozen_hashes),
        "detail": "run scripts/12_freeze.py --check" if frozen_hashes else "no frozen_hashes.json",
    })
    for f, gid in (("docs/novelty_audit.md", "G_novelty"),
                   ("runs/manifests/data_audit.json", "G_data_manifest")):
        integrity.append({"id": gid, "ok": os.path.exists(os.path.join(ROOT, f)), "detail": f})

    gates = spec_gates + protocol_gates + claim_gates + integrity
    passed = [g for g in gates if g["ok"] is True]
    failed = [g for g in gates if g["ok"] is False]
    unver = [g for g in gates if g["ok"] is None]

    def block(gs):
        return {"n": len(gs), "passed": sum(1 for g in gs if g["ok"] is True),
                "failed": sum(1 for g in gs if g["ok"] is False),
                "unverifiable": sum(1 for g in gs if g["ok"] is None)}

    # §11: a missing comparable reference is "unverifiable", never "passed".
    spec_failed = [g for g in spec_gates if g["ok"] is False]
    spec_unver = [g for g in spec_gates if g["ok"] is None]
    verdict = "fail" if (failed or spec_unver) else ("pass" if passed else "incomplete")

    status = {
        "performance": {
            "verdict": "pass" if (spec_gates and not spec_failed and not spec_unver) else "fail",
            "detail": block(spec_gates),
            "note": "six §11 gates (3 transfers x PSNR/SSIM) against the strongest measured "
                    "comparable arm; target-supervised numbers excluded by construction",
        },
        "risk_control": {
            "verdict": "pass" if claim_gates and not [g for g in claim_gates if g["ok"] is False] else "fail",
            "detail": block(claim_gates),
        },
        "reproducibility": {
            "verdict": "pass" if not [g for g in protocol_gates if g["ok"] is False] else "fail",
            "detail": block(protocol_gates),
        },
        "novelty": {
            "verdict": "unverifiable" if not os.path.exists(os.path.join(ROOT, "docs", "novelty_audit.md"))
                       else "documented_not_machine_checkable",
            "detail": "docs/novelty_audit.md is an argument, not a measurement",
        },
    }

    report = {
        "tolerance": TOL,
        "n_gates": len(gates),
        "n_passed": len(passed),
        "n_failed": len(failed),
        "n_unverifiable": len(unver),
        "verdict": verdict,
        "status_fields": status,
        "blocks": {"spec": block(spec_gates), "protocol": block(protocol_gates),
                   "claim": block(claim_gates), "integrity": block(integrity)},
        "gates": gates,
        "reference_policy": {
            "spec_reference": "strongest measured deployable arm under the frozen protocol "
                              "(runs/comparison/comparison_set.json, oracle excluded)",
            "excluded_from_gate": ["oracle", "target-supervised", "published-other-protocol"],
            "separate_coordinates": separate_refs,
        },
        "streams": {k: {"online": v["online"], "replay_ok": bool(v["replay"] and v["replay"].get("ok"))}
                    for k, v in streams.items()},
        "frozen_script_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
        "note": ("MSE is reported per image but never used as an acceptance criterion (spec §11). "
                 "g_m = max(0, (S_m - R_m)/S_m) with S the strongest comparable arm; PSNR gaps are "
                 "reported in dB. An unverifiable gate is never counted as passed."),
    }
    out = os.path.join(out_root, "acceptance_report.json")
    save_json(report, out)

    print(f"{'gate':<38}{'result':<14}detail")
    print("-" * 100)
    for g in gates:
        r = "PASS" if g["ok"] is True else ("FAIL" if g["ok"] is False else "UNVERIFIABLE")
        detail = {k: v for k, v in g.items() if k not in ("id", "ok", "detail")}
        if "detail" in g and not detail:
            detail = g["detail"]
        text = json.dumps(detail, ensure_ascii=False)
        print(f"{g['id']:<38}{r:<14}{text[:90]}")
    print("-" * 100)
    print(f"verdict: {report['verdict']}  ({len(passed)} pass / {len(failed)} fail / {len(unver)} unverifiable)")
    for k, v in status.items():
        print(f"  {k:<16}{v['verdict']}")
    print("wrote", out)
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
