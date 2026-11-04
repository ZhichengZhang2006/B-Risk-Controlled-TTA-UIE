#!/usr/bin/env bash
# gpu_run.sh -- serialise GPU work on this pod.
#
# MEASURED BEHAVIOUR (this is not paranoia, it is a hardware constraint here):
# a SECOND CUDA process cannot initialise while the first one holds the device.
# The pod's libcuda.so.1 is a shim (libcudawrapper.so.1) that hands the GPU to
# one process; every other process gets
#     cuInit -> 4   (CUDA_ERROR_INSUFFICIENT_DRIVER)
#     torch.cuda.is_available() -> False
#     "Unexpected error from cudaGetDeviceCount() ... Error 4: driver shutting down"
# Observed while `01b_adapter_pretrain.py` was happily training at 100 s/epoch:
# `00c_metric_validation.py` crashed inside mamba_ssm with
#     RuntimeError: Expected u.is_cuda() to be true, but got false
# which is a *secondary* symptom -- the real cause is the occupied slot.
#
# So: never launch two GPU jobs at once.  This wrapper takes a flock and waits
# its turn, which makes "run everything in sequence" the default rather than a
# discipline every caller has to remember.
#
# Usage:
#   bash scripts/gpu_run.sh 04h_theta0_eval.py --overfit 8 --steps 240
#   DEV=1 bash scripts/gpu_run.sh 05_online_eval.py --stream LSUI2UIEB
#   LOCK_TIMEOUT=7200 bash scripts/gpu_run.sh <script> [args...]
#
# The exit code is the wrapped command's exit code, so callers can still branch.

set -uo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: bash scripts/gpu_run.sh <script-under-scripts/> [args...]" >&2
  exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
PY="${PY:-/home/user/Conda/CDF_FFT/bin/python}"
DEV="${DEV:-0}"
LOCK="${GPU_LOCK:-/tmp/brctta_gpu.lock}"
LOCK_TIMEOUT="${LOCK_TIMEOUT:-14400}"

SCRIPT="$1"; shift
if [ ! -f "$HERE/$SCRIPT" ] && [ ! -f "$SCRIPT" ]; then
  echo "[gpu_run] no such script: $SCRIPT" >&2
  exit 2
fi
TARGET="$HERE/$SCRIPT"
[ -f "$TARGET" ] || TARGET="$SCRIPT"

cd "$ROOT" || exit 2
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$DEV"
export BRCTTA_TORCH_THREADS="${BRCTTA_TORCH_THREADS:-4}"

echo "[gpu_run] $(date -Is) waiting for $LOCK ..."
exec 9>"$LOCK"
if ! flock -w "$LOCK_TIMEOUT" 9; then
  echo "[gpu_run] timed out after ${LOCK_TIMEOUT}s waiting for the GPU lock" >&2
  exit 75
fi
echo "[gpu_run] $(date -Is) lock acquired, running $SCRIPT $*"

# The flock only orders OUR jobs.  Other tenants on this pod hold the same
# single CUDA slot through a different lock we cannot see, so the lock being
# free does NOT imply the driver is usable.  Probe cuInit directly and wait for
# it to return 0 (CUDA_SUCCESS).  Without this the job starts, torch sees
# is_available()==False, and mamba_ssm dies with the misleading
# "Expected u.is_cuda() to be true".
GPU_WAIT="${GPU_WAIT:-7200}"       # seconds to keep retrying the driver
GPU_POLL="${GPU_POLL:-20}"         # seconds between probes
_probe_gpu() {
  "$PY" - <<'EOF' >/dev/null 2>&1
import ctypes
try:
    raise SystemExit(0 if ctypes.CDLL("libcuda.so.1").cuInit(0) == 0 else 1)
except OSError:
    raise SystemExit(1)
EOF
}
waited=0
until _probe_gpu; do
  if [ "$waited" -ge "$GPU_WAIT" ]; then
    echo "[gpu_run] $(date -Is) GPU unusable after ${GPU_WAIT}s (cuInit != 0); giving up" >&2
    exit 76
  fi
  [ "$waited" = 0 ] && echo "[gpu_run] $(date -Is) CUDA slot busy (cuInit != 0), waiting up to ${GPU_WAIT}s ..."
  sleep "$GPU_POLL"
  waited=$((waited + GPU_POLL))
done
[ "$waited" -gt 0 ] && echo "[gpu_run] $(date -Is) CUDA slot free after ${waited}s"

if command -v nvidia-smi >/dev/null 2>&1; then
  BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -vcE '^\s*$' || true)
  [ "${BUSY:-0}" -gt 0 ] && echo "[gpu_run] note: $BUSY CUDA process(es) resident, util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | head -1)"
fi

"$PY" -u "$TARGET" "$@" 2>&1 | sed -u '/cannot find schd/d'
rc="${PIPESTATUS[0]}"
echo "[gpu_run] $(date -Is) $SCRIPT rc=$rc"
exit "$rc"
