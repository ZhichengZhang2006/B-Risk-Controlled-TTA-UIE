#!/bin/bash
# Pre-train the residual adapter on each source domain's own pairs.
#
# theta_0 = frozen source backbone + adapter trained on SOURCE pairs only.
# No target image/label/statistic is touched here, so theta_0 stays a
# source-only model under spec §5.1 -- it is simply a better one than the bare
# backbone.  TTA gain is measured against `source_adapter` (theta_0, no online
# update), never against the bare backbone.
#
# NOTE: `sed -u` (unbuffered) rather than grep on purpose.  grep block-buffers
# when stdout is a file, which hides every epoch line for minutes and makes a
# healthy run look hung.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="${PY:-/home/user/Conda/CDF_FFT/bin/python}"

for spec in "LSUI 6" "UIEB 10"; do
  set -- $spec
  ds=$1; ep=$2
  echo "=== pretrain $ds ${ep} epochs $(date -Is)"
  "$PY" -u scripts/01b_adapter_pretrain.py --source "$ds" --epochs "$ep" --backbone mambacue \
      2>&1 | sed -u '/cannot find schd/d'
  echo "=== pretrain $ds rc=${PIPESTATUS[0]} $(date -Is)"
done
echo "=== adapter pretrain done $(date -Is)"
