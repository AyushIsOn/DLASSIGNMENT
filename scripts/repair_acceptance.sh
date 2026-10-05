#!/usr/bin/env bash
# Run from the repaired Git checkout, using the existing runtime's venv/artifacts.
set -euo pipefail
SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${1:?usage: repair_acceptance.sh WORKSPACE cpu|gpu}"
MODE="${2:?choose cpu or gpu}"
WORKSPACE="$(cd "$WORKSPACE" && pwd)"
PYTHON="$WORKSPACE/.venv/bin/python"
export PYTHONPATH="$SOURCE/src:$SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
cd "$SOURCE"
if [[ "$MODE" == cpu ]]; then
  export CUDA_VISIBLE_DEVICES=""
  "$PYTHON" -u -m acharya.cli evaluate --workspace "$WORKSPACE" \
    --retrieval-mode full --golden "$WORKSPACE/eval/golden.jsonl" --activate-calibration
  "$PYTHON" -u -m acharya.cli calibrate-support --workspace "$WORKSPACE"
  echo 'CPU recalibration complete. GPU acceptance is still required.'
elif [[ "$MODE" == gpu ]]; then
  "$PYTHON" -u -m training.acceptance --workspace "$WORKSPACE" \
    --adapter "$WORKSPACE/artifacts/external-run/adapter" \
    --output "$WORKSPACE/artifacts/external-run/evaluation/acceptance-repair.json"
else
  echo 'mode must be cpu or gpu' >&2
  exit 2
fi
