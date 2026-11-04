"""Produce the figures required by spec §13.

Three figures, each answering one question the report must answer:

  1. `curves_<stream>.png` -- PSNR/SSIM of the running state over the stream for
     every arm.  Shows *when* an arm gains or loses, which a single number hides.
  2. `dpsnr_hist_<stream>.png` -- per-image ΔPSNR histogram (post minus pre) for
     the gate.  The negative-transfer story lives in the left tail.
  3. `cer_<stream>.png` -- cumulative ΔPSNR vs stream position: the gate's
     cumulative regret against frozen and against always-accept.

No numbers are invented here: every panel is read from an action log.  If a log
is missing the figure is skipped and the reason is printed.
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import load_config, run_dir  # noqa: E402

COLORS = {"source_frozen": "#888888", "source_adapter": "#444444",
          "always_accept": "#d95f02", "random_gate": "#7570b3",
          "oracle": "#1b9e77", "simple_gate": "#e7298a",
          "stochastic": "#66a61e", "ours": "#e41a1c"}


def read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def find_logs(cfg, stream):
    """All action logs for one stream, keyed by method name."""
    logs = {}
    for path in glob.glob(os.path.join(cfg["paths"]["out_root"], "online_*", "action_log_*.jsonl")):
        base = os.path.basename(path)
        method = base[len("action_log_"):-len(".jsonl")]
        d = json.load(open(os.path.join(os.path.dirname(path), "online_results.json")))
        if d.get("stream") != stream:
            continue
        logs[method] = path
    return logs


def running_mean(v, k=15):
    if len(v) < k:
        return np.array(v, dtype=np.float64)
    c = np.cumsum(np.insert(np.array(v, dtype=np.float64), 0, 0.0))
    out = (c[k:] - c[:-k]) / k
    return np.concatenate([np.full(k - 1, out[0]), out])


def plot_curves(logs, out_png, stream):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for method, path in sorted(logs.items()):
        recs = read_jsonl(path)
        if not recs:
            continue
        psnr = [r.get("psnr_pre", np.nan) for r in recs]
        ssim = [r.get("ssim_pre", np.nan) for r in recs]
        c = COLORS.get(method, None)
        axes[0].plot(running_mean(psnr), label=method, color=c, lw=1.4)
        axes[1].plot(running_mean(ssim), label=method, color=c, lw=1.4)
    for ax, ylab in zip(axes, ("PSNR (dB, prequential)", "SSIM (prequential)")):
        ax.set_xlabel("stream position t")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"{stream}: running state quality over the stream (window 15)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def plot_hist(logs, out_png, stream, method="ours"):
    path = logs.get(method)
    if not path:
        return None
    recs = read_jsonl(path)
    d = np.array([r.get("psnr_post", np.nan) - r.get("psnr_pre", np.nan) for r in recs])
    d = d[np.isfinite(d)]
    if not len(d):
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.hist(d, bins=45, color="#4c72b0", alpha=0.85)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(float(np.mean(d)), color="#e41a1c", ls="--", lw=1.2,
               label=f"mean {np.mean(d):+.3f} dB")
    ax.set_xlabel("per-image ΔPSNR = post − pre (dB)")
    ax.set_ylabel("images")
    ax.set_title(f"{stream}: {method} per-image change\nNAR0={float((d < 0).mean()):.3f}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return {"mean": float(np.mean(d)), "nar0": float((d < 0).mean()), "n": int(len(d))}


def plot_cer(logs, out_png, stream):
    """Cumulative (running-mean) ΔPSNR relative to the frozen arm."""
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for method, path in sorted(logs.items()):
        recs = read_jsonl(path)
        if not recs:
            continue
        psnr = [r.get("psnr_pre", np.nan) for r in recs]
        ax.plot(running_mean(psnr), label=method, color=COLORS.get(method), lw=1.5)
    ax.set_xlabel("stream position t")
    ax.set_ylabel("running-mean PSNR of the committed state (dB)")
    ax.set_title(f"{stream}: prequential quality (what the user actually sees)", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="", help="default: every stream with logs")
    args = ap.parse_args()
    cfg = load_config()

    streams = [args.stream] if args.stream else sorted(
        json.load(open(p)).get("stream", "")
        for p in glob.glob(os.path.join(cfg["paths"]["out_root"], "online_*", "online_results.json"))
    )
    streams = [s for s in dict.fromkeys(streams) if s]
    if not streams:
        print("[plots] no online results yet")
        return 1

    made = []
    for stream in streams:
        logs = find_logs(cfg, stream)
        if not logs:
            print(f"[plots] {stream}: no logs")
            continue
        out = run_dir(cfg, "figures")
        p1 = os.path.join(out, f"curves_{stream}.png")
        plot_curves(logs, p1, stream)
        p2 = os.path.join(out, f"dpsnr_hist_{stream}.png")
        stats = plot_hist(logs, p2, stream)
        p3 = os.path.join(out, f"prequential_{stream}.png")
        plot_cer(logs, p3, stream)
        made += [p1, p3] + ([p2] if stats else [])
        print(f"[plots] {stream}: {len(logs)} arms -> {p1}")

    print(f"[plots] {len(made)} figures in {run_dir(cfg, 'figures')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
