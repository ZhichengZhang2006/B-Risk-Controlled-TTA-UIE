#!/bin/bash
# Sequential launcher: several CUDA processes initialising at once crashed with
# "Error 4: driver shutting down" on this shared pod, so run them one at a time.
set -u
cd /home/user/BRCTTA
export PYTHONPATH=/home/user/BRCTTA/src
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
PY=/home/user/Conda/CDF_FFT/bin/python

for S in "$@"; do
  echo "=== $(date -Is) selecting adapter for $S"
  $PY scripts/04b_select_adapt.py --stream "$S" --n 60 --dev-domain target \
      --grid 1x0.001,3x0.003,5x0.01
  echo "=== $(date -Is) finished $S (exit $?)"
done
echo "ALL_SELECTIONS_DONE"
