"""Fit the underwater degradation model ``D: clean -> degraded`` on SOURCE pairs.

Training direction matters
--------------------------
Source split gives us ``(x_s, g_s) = (degraded input, clean GT)``.  At test time
we want to ask "if my output ``y`` were the clean image of this scene, would
re-degrading it reproduce the observed input ``x``?".  So ``D`` must map
**clean -> degraded**, i.e. we fit ``D(g_s) ≈ x_s``.

(Running it the other way, ``D(x_s) ≈ g_s``, would fit an *enhancer*, and then
``||D(y) - x||`` would be measuring "does my output look like a degraded input",
which is the opposite of what we want.)

Usage
-----
    python scripts/00e_degrade_train.py --source LSUI --epochs 12
    python scripts/00e_degrade_train.py --source UIEB --epochs 20 --limit 400

Writes ``runs/degrader_<SOURCE>/degrader.pth`` plus a json with the loss history
and the sha256 of the weights, so the risk pipeline can cite exactly which
degradation model produced which evidence.
"""

import argparse
import hashlib
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from brctta.degrade import Degrader, redegradation_loss  # noqa: E402
from brctta.data import StreamSpec  # noqa: E402

from common import load_config, open_stream, run_dir, save_json  # noqa: E402


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="LSUI | UIEB | UFO | EUVPd")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="0 = all source train pairs")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = load_config()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("[warn] running on CPU; the pod's single CUDA slot is probably busy")

    spec = StreamSpec(
        name=f"degrade_{args.source}",
        source_dataset=args.source,
        target_split="train",
        seed=args.seed,
        order="shuffle",
    )
    stream = open_stream(cfg, args.source, "train", spec)
    n = len(stream)
    if args.limit:
        n = min(n, args.limit)
    print(f"[degrader] source={args.source} pairs={n} (of {len(stream)}) device={device}")

    net = Degrader(width=args.width).to(device)
    nparam = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, args.epochs * ((n + args.batch - 1) // args.batch)),
        pct_start=0.15,
    )

    history = []
    t0 = time.time()
    for ep in range(args.epochs):
        order = torch.randperm(n).tolist()
        acc, seen = 0.0, 0
        for i in range(0, n, args.batch):
            idxs = order[i:i + args.batch]
            # CachedStream.take_batch -> (xs, ys, stems): xs = degraded input,
            # ys = clean GT (the cache stores the protocol tensors verbatim).
            x, g = stream.take_batch(idxs)[:2]
            x = x.to(device)
            g = g.to(device)  # clean
            # D(clean) should reproduce the degraded input
            loss = redegradation_loss(net, g, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            acc += float(loss) * len(idxs)
            seen += len(idxs)
        mean = acc / max(1, seen)
        history.append({"epoch": ep, "loss": mean, "seconds": round(time.time() - t0, 1)})
        print(f"  epoch {ep:2d} redegrad_loss {mean:.6f}  ({time.time() - t0:.0f}s)")

    out = run_dir(cfg, f"degrader_{args.source}{args.tag}")
    os.makedirs(out, exist_ok=True)
    ckpt = os.path.join(out, "degrader.pth")
    torch.save({"degrader": net.state_dict(), "source": args.source, "width": args.width,
                "epochs": args.epochs, "limit": args.limit, "loss_history": history}, ckpt)
    payload = {
        "source": args.source, "width": args.width, "params": nparam,
        "pairs": n, "epochs": args.epochs, "final_loss": history[-1]["loss"] if history else None,
        "first_loss": history[0]["loss"] if history else None,
        "ckpt": ckpt, "sha256": sha256_file(ckpt), "history": history,
        "direction": "D(clean GT) -> degraded input; residual is ||D(y) - x||_1",
    }
    save_json(payload, os.path.join(out, "degrader.json"))
    print(f"[degrader] saved {ckpt} sha256={payload['sha256'][:16]} loss {payload['first_loss']:.6f} -> {payload['final_loss']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
