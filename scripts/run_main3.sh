#!/usr/bin/env bash
# Run the three locked transfers end-to-end on the MambaCue backbone, then rebuild
# the §11 comparison set and the acceptance report.
#
#   bash scripts/run_main3.sh            # all three, sequential
#   STREAMS="LSUI2UIEB" bash scripts/run_main3.sh
#
# Sequential on purpose: the pod exposes a single CUDA slot, so two concurrent
# CUDA processes make the second one fail with a misleading
# "Expected u.is_cuda() to be true" from inside mamba_ssm.
set -uo pipefail

PY=${PY:-/home/user/Conda/CDF_FFT/bin/python}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
STREAMS=${STREAMS:-"LSUI2UIEB UIEB2LSUI LSUI2UFO"}
BACKBONE=${BACKBONE:-mambacue}

echo "=================== $(date -Is) main table, backbone=$BACKBONE ==================="
for S in $STREAMS; do
  echo "############ $S $(date -Is)"
  bash scripts/run_stream.sh "$S" "$BACKBONE" target
  rc=$?
  echo "############ $S rc=$rc $(date -Is)"
  if [[ $rc -ne 0 ]]; then
    echo "[run_main3] stream $S FAILED rc=$rc; continuing to next stream" >&2
  fi
done

echo "--- [08] comparison set + acceptance"
$PY scripts/08_comparison.py || echo "[run_main3] comparison failed"
$PY scripts/08_acceptance.py || echo "[run_main3] acceptance failed"
echo "--- [10] ledger"
$PY scripts/10_ledger.py || echo "[run_main3] ledger failed"
echo "--- [11] figures"
$PY scripts/11_plots.py || echo "[run_main3] plots failed"
echo "=================== $(date -Is) done ==================="
