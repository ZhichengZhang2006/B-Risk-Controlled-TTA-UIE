"""08b -- render ``runs/acceptance_report.json`` as the human-readable
``runs/acceptance_report.md`` required by spec §13.

Deliberately a separate script: the acceptance *decision* lives in
``08_acceptance.py`` and is hashed into the report.  This file only formats the
already-decided numbers, so re-rendering cannot change a verdict.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config  # noqa: E402


def fmt(v, nd=4):
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int,)):
        return str(v)
    if isinstance(v, float):
        if v != v:
            return "n/a"
        if v == float("inf"):
            return "+inf"
        if v == float("-inf"):
            return "-inf"
        return f"{v:.{nd}f}"
    return "" if v is None else str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = cfg["paths"]["out_root"]
    jp = os.path.join(root, "acceptance_report.json")
    if not os.path.exists(jp):
        print("[08b] no acceptance_report.json; run 08_acceptance.py first")
        return 2
    with open(jp, encoding="utf-8") as fh:
        r = json.load(fh)

    L = []
    L.append("# Acceptance report (§11)\n")
    L.append(f"**Verdict: `{r['verdict']}`** — {r['n_passed']} pass / {r['n_failed']} fail / "
             f"{r['n_unverifiable']} unverifiable (of {r['n_gates']}).\n")
    L.append(f"Relative-shortfall tolerance: **{r['tolerance']:.0%}**. "
             f"Script SHA256: `{r['frozen_script_sha256'][:16]}…`\n")

    L.append("\n## Status fields\n")
    L.append("The four axes are reported separately and must not be collapsed into one score.\n")
    L.append("| field | verdict | detail |\n| --- | --- | --- |")
    for k, v in r["status_fields"].items():
        d = v.get("detail")
        if isinstance(d, dict):
            d = f"{d.get('passed', 0)}/{d.get('n', 0)} passed, {d.get('failed', 0)} failed, {d.get('unverifiable', 0)} unverifiable"
        L.append(f"| {k} | `{v['verdict']}` | {d} |")

    L.append("\n## Gates\n")
    L.append("| gate | result | detail |\n| --- | --- | --- |")
    for g in r["gates"]:
        res = "PASS" if g["ok"] is True else ("FAIL" if g["ok"] is False else "UNVERIFIABLE")
        detail = {k: v for k, v in g.items() if k not in ("id", "ok")}
        if "detail" in detail and len(detail) == 1:
            text = detail["detail"]
            text = "" if text is None else (text if isinstance(text, str) else json.dumps(text))
        else:
            detail.pop("detail", None)
            text = "; ".join(f"{k}={fmt(v)}" for k, v in detail.items())
        L.append(f"| `{g['id']}` | {res} | {text[:220]} |")

    L.append("\n## Reference policy\n")
    rp = r["reference_policy"]
    L.append(f"- gate reference: {rp['spec_reference']}")
    L.append(f"- excluded from every gate: {', '.join('`%s`' % x for x in rp['excluded_from_gate'])}")
    L.append("\nTarget-supervised and published numbers, reported as coordinates only:\n")
    for name, blk in (rp.get("separate_coordinates") or {}).items():
        L.append(f"- **{name}**: {json.dumps(blk, ensure_ascii=False)}")

    L.append("\n## Note\n")
    L.append(r["note"] + "\n")

    out = os.path.join(root, "acceptance_report.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print(f"[08b] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
