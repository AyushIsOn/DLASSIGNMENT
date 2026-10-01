#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=""
QUOTE=""
TRAIN_STEPS=100
MERGE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace) WORKSPACE="$2"; shift 2 ;;
    --quote) QUOTE="$2"; shift 2 ;;
    --train-steps) TRAIN_STEPS="$2"; shift 2 ;;
    --merge) MERGE=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$WORKSPACE" && -n "$QUOTE" ]] || {
  echo "--workspace and --quote are required" >&2
  exit 2
}
[[ "$TRAIN_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "invalid --train-steps" >&2; exit 2; }
export PYENV_VERSION="${PYENV_VERSION:-3.11.15}"
export ACHARYA_PERSISTENT_STORAGE="${ACHARYA_PERSISTENT_STORAGE:-0}"
export ACHARYA_AUTO_STOP="${ACHARYA_AUTO_STOP:-0}"
RUN_ROOT="$WORKSPACE/artifacts/external-run"
mkdir -p "$RUN_ROOT"
STARTED=$SECONDS

run_stage() {
  local name="$1"
  local cap_minutes="$2"
  shift 2
  local remaining=$((14400 - (SECONDS - STARTED)))
  local stage_cap=$((cap_minutes * 60))
  if (( remaining <= 0 )); then
    echo "overall 240-minute cap reached before $name" >&2
    exit 3
  fi
  if (( stage_cap > remaining )); then stage_cap=$remaining; fi
  echo "[$name] cap=${stage_cap}s"
  timeout --signal=TERM --kill-after=30s "$stage_cap" \
    uv run --project "$WORKSPACE" python -m acharya.lightning run-stage \
    --workspace "$WORKSPACE" --name "$name" -- "$@"
}

# This checks quote conversion, one A100 >=39 GiB, disk, persistent storage,
# auto-stop, preparation hashes, and every clean-extraction bundle hash before model load.
run_stage preflight 10 uv run --project "$WORKSPACE" python -m acharya.lightning preflight \
  --workspace "$WORKSPACE" --quote "$QUOTE"
run_stage activate 10 uv run --project "$WORKSPACE" python -m acharya.bundle activate \
  --workspace "$WORKSPACE"
run_stage model_acquisition 30 uv run --project "$WORKSPACE" python \
  "$WORKSPACE/training/train_qlora.py" --workspace "$WORKSPACE" --acquire-only

# A five-step real smoke is bounded separately; the 100-step run is the paid profile/checkpoint.
run_stage smoke 20 uv run --project "$WORKSPACE" python "$WORKSPACE/training/train_qlora.py" \
  --workspace "$WORKSPACE" --output "$RUN_ROOT/smoke" --max-steps 5
run_stage profile 45 uv run --project "$WORKSPACE" python "$WORKSPACE/training/train_qlora.py" \
  --workspace "$WORKSPACE" --output "$RUN_ROOT/checkpoint" --max-steps 100
if (( TRAIN_STEPS > 100 )); then
  run_stage training 240 uv run --project "$WORKSPACE" python "$WORKSPACE/training/train_qlora.py" \
    --workspace "$WORKSPACE" --output "$RUN_ROOT/checkpoint" --max-steps "$TRAIN_STEPS"
fi
run_stage evaluation 45 uv run --project "$WORKSPACE" python "$WORKSPACE/training/evaluate_adapter.py" \
  --workspace "$WORKSPACE" --adapter "$RUN_ROOT/checkpoint/adapter" \
  --output "$RUN_ROOT/evaluation/evaluation.json"
run_stage export 30 uv run --project "$WORKSPACE" python "$WORKSPACE/training/export_adapter.py" \
  --workspace "$WORKSPACE" --source "$RUN_ROOT/checkpoint/adapter" \
  --destination "$RUN_ROOT/adapter"
if (( MERGE == 1 )); then
  run_stage merge 45 uv run --project "$WORKSPACE" python "$WORKSPACE/training/merge_adapter.py" \
    --workspace "$WORKSPACE" --adapter "$RUN_ROOT/adapter" \
    --destination "$RUN_ROOT/merged"
fi

# Rebuild retrieval from the bundled prepared corpus; generation stays behind support calibration.
run_stage rag_index 45 uv run --project "$WORKSPACE" acharya --workspace "$WORKSPACE" build-index
run_stage rag_calibration 45 uv run --project "$WORKSPACE" acharya --workspace "$WORKSPACE" calibrate
run_stage support_calibration 45 uv run --project "$WORKSPACE" acharya --workspace "$WORKSPACE" calibrate-support
run_stage rag_smoke 45 uv run --project "$WORKSPACE" python "$WORKSPACE/training/peft_rag_smoke.py" \
  --workspace "$WORKSPACE" --adapter "$RUN_ROOT/adapter" \
  --output "$RUN_ROOT/serving/peft-rag-smoke.json"
run_stage finalize 10 uv run --project "$WORKSPACE" python -m acharya.lightning finalize \
  --workspace "$WORKSPACE" --run-root "$RUN_ROOT"
