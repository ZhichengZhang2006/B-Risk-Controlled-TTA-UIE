#!/usr/bin/env bash
# End-to-end pipeline for one locked transfer (spec §5.1).
#
#   bash scripts/run_stream.sh LSUI2UIEB                 # mambacue backbone
#   bash scripts/run_stream.sh LSUI2UIEB aqua            # strong frozen backbone
#
# Order matters and is enforced here: adapter budget and the risk controller are
# fitted on the DEV split only; the test split first becomes visible in step 05.
# No step writes anything based on a test-side number.
set -euo pipefail

STREAM=${1:?stream name}
BACKBONE=${2:-mambacue}
DEV=${3:-target}
PY=${PY:-/home/user/Conda/CDF_FFT/bin/python}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1

SUF=""
[[ "$BACKBONE" != "mambacue" ]] && SUF="_${BACKBONE}"

LOG=logs_${STREAM}${SUF}.log
exec > >(tee -a "$LOG") 2>&1
echo "=================== $(date -Is) stream=$STREAM backbone=$BACKBONE ==================="

# 04b/04c write runs/selection_<STREAM>_<PROBE_DATASET><SUF>/, i.e. the probe
# dataset name rather than the literal word "target".
resolve_sel() {
  local target_ds
  target_ds=$($PY -c "import yaml;c=yaml.safe_load(open('configs/main.yaml'));print(next(s for s in c['streams'] if s['name']=='${STREAM}')['target'])")
  for c in "runs/selection_${STREAM}_${target_ds}${SUF}" "runs/selection_${STREAM}_${DEV}${SUF}"; do
    [[ -f "$c/adapt_selection.json" ]] && { echo "$c/adapt_selection.json"; return 0; }
  done
  return 1
}

if ! SEL=$(resolve_sel); then
  echo "--- [04b] selecting adapter budget on dev (${DEV})"
  bash scripts/gpu_run.sh 04b_select_adapt.py --stream "$STREAM" --n 60 --dev-domain "$DEV" \
      --backbone "$BACKBONE" --grid 1x0.001,3x0.003,5x0.01
  SEL=$(resolve_sel)
fi
read -r STEPS LR < <($PY -c "import json;d=json.load(open('$SEL'));print(d['selected_steps'],d['selected_lr'])")
export SEL_FILE="$SEL"
echo "adapter budget: steps=$STEPS lr=$LR  (from $SEL)"

echo "--- [02] episodes on dev"
bash scripts/gpu_run.sh 02_episodes.py --stream "$STREAM" --dev-domain "$DEV" --episode-len 32 \
    --steps "$STEPS" --lr "$LR" --backbone "$BACKBONE"
# 02 names the directory after which dev domain it actually used, so resolve it
# from disk rather than reconstructing the name here.
EPDIR=$(ls -td runs/episodes_${STREAM}_*dev*${SUF} 2>/dev/null | head -1)
EPFILE="$EPDIR/episodes.jsonl"
if [[ ! -f "$EPFILE" ]]; then
  echo "[fatal] no episodes at $EPFILE" >&2
  exit 1
fi
echo "episodes: $EPFILE"

echo "--- [03] risk controller"
$PY scripts/03_risk_train.py --stream "$STREAM" --episodes "$EPFILE"

echo "--- [05] causal online evaluation on the locked test split"
bash scripts/gpu_run.sh 05_online_eval.py --stream "$STREAM" --steps "$STEPS" --lr "$LR" --backbone "$BACKBONE"

echo "--- [07] replay check"
bash scripts/gpu_run.sh 07_replay.py --stream "$STREAM" --steps "$STEPS" --lr "$LR" --backbone "$BACKBONE"

echo "=================== $(date -Is) done stream=$STREAM ==================="
