#!/usr/bin/env bash
# Run the three locked transfers (spec §5.1) end to end on the STRONG backbone.
#
# Why the strong backbone: §11 requires the comparison set to contain both a
# strong frozen model and a strong TTA, and forbids letting one weak backbone
# stand in for the whole set.  Measuring the risk-controlled adapter on top of
# the weak MambaCue backbone says nothing about whether the mechanism survives
# on a strong model; on AquaClarity it does, because theta_0 *is* the strong
# frozen baseline, so any accepted update has to beat it.
#
# Everything GPU goes through scripts/gpu_run.sh (one CUDA slot per pod).
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PY=${PY:-/home/user/Conda/CDF_FFT/bin/python}
GRID=${GRID:-1x0.001,3x0.003,5x0.01}
N=${N:-48}
LOG=logs_all_aqua.log
exec > >(tee -a "$LOG") 2>&1

echo "=================== $(date -Is) all streams, backbone=aqua ==================="

for STREAM in LSUI2UIEB UIEB2LSUI LSUI2UFO; do
  echo "############ $STREAM $(date -Is)"

  # --- dev-only: pick the adapter budget under the fixed dev criterion
  if ! ls -d runs/selection_${STREAM}_*_aqua >/dev/null 2>&1; then
    bash scripts/gpu_run.sh 04b_select_adapt.py --stream "$STREAM" --n "$N" \
        --dev-domain target --backbone aqua --grid "$GRID"
  fi
  SEL=$(ls -d runs/selection_${STREAM}_*_aqua 2>/dev/null | head -1)/adapt_selection.json
  if [[ ! -f "$SEL" ]]; then echo "[fatal] no selection for $STREAM"; continue; fi
  read -r STEPS LR < <($PY -c "import json;d=json.load(open('$SEL'));print(d['selected_steps'],d['selected_lr'])")
  echo "--- $STREAM selected ${STEPS}x${LR} from $SEL"

  # --- dev-only: episodes for the risk controller
  bash scripts/gpu_run.sh 02_episodes.py --stream "$STREAM" --dev-domain target \
      --episode-len 32 --steps "$STEPS" --lr "$LR" --backbone aqua || continue
  EPDIR=$(ls -td runs/episodes_${STREAM}_*dev*_aqua 2>/dev/null | head -1)
  EPFILE="$EPDIR/episodes.jsonl"
  if [[ ! -f "$EPFILE" ]]; then echo "[fatal] no episodes at $EPFILE"; continue; fi

  bash scripts/gpu_run.sh 03_risk_train.py --stream "$STREAM" --episodes "$EPFILE" || continue

  # --- the locked test split becomes visible only here
  bash scripts/gpu_run.sh 05_online_eval.py --stream "$STREAM" --steps "$STEPS" --lr "$LR" \
      --backbone aqua || continue
  bash scripts/gpu_run.sh 07_replay.py --stream "$STREAM" --steps "$STEPS" --lr "$LR" \
      --backbone aqua || continue
done

echo "--- comparison set + acceptance"
$PY scripts/08_comparison.py
$PY scripts/08_acceptance.py
echo "=================== $(date -Is) done ==================="
