#!/usr/bin/env bash
# STEP 2 - run on the H200 after scripts/lightning_cpu_setup.sh finished on CPU.
#
#   bash scripts/lightning_gpu_run.sh
#
# preflight -> train (~1.5-2 h for 14B) -> evaluate base vs fine-tuned (~25 min) -> report
# -> merge. (GGUF for the Mac is a separate CPU step afterwards: scripts/export_gguf.sh.) Runs in its own background session: closing the
# browser or pressing Ctrl+C only stops the log view. Run the same command again to
# re-attach, or - if the Studio was stopped - to resume: finished stages are skipped and
# training continues from its last checkpoint.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
PIDFILE="$ROOT/logs/gpu_run.pid"

if [[ "${ACHARYA_DETACHED:-0}" != "1" ]]; then
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "A GPU run is already in progress (pid $(cat "$PIDFILE")). Showing its log:"
    exec tail -n 60 -f "$(cat "$ROOT/logs/gpu_run.current")" --pid="$(cat "$PIDFILE")"
  fi
  LOG="$ROOT/logs/gpu_run_$(date +%Y%m%d_%H%M%S).log"
  echo "$LOG" > "$ROOT/logs/gpu_run.current"
  # setsid: own session, so Ctrl+C / closing the terminal never reaches the run
  ACHARYA_DETACHED=1 setsid nohup bash "$ROOT/scripts/lightning_gpu_run.sh" "$@" \
    > "$LOG" 2>&1 < /dev/null &
  echo $! > "$PIDFILE"
  echo "Started in the background (pid $!). Full log: $LOG"
  echo "Following the log - Ctrl+C stops FOLLOWING only, the run keeps going."
  echo "Re-attach any time with: bash scripts/lightning_gpu_run.sh"
  sleep 2
  exec tail -n +1 -f "$LOG" --pid="$(cat "$PIDFILE")"
fi

source scripts/_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
started=$(date +%s)
# GPU budget: the whole run (setup + training + evaluation + merge) must fit in
# ACHARYA_GPU_MINUTES. Training stops early if needed and keeps EVAL_RESERVE minutes for the rest.
GPU_MINUTES="${ACHARYA_GPU_MINUTES:-105}"
EVAL_RESERVE="${ACHARYA_EVAL_RESERVE_MINUTES:-30}"
export ACHARYA_TRAIN_DEADLINE=$(( started + (GPU_MINUTES - EVAL_RESERVE) * 60 ))
stop_studio() {  # ACHARYA_STOP_WHEN_DONE=1: stop the Studio (and the GPU bill) when finished
  [[ "${ACHARYA_STOP_WHEN_DONE:-0}" == "1" ]] || return 0
  echo "Stopping this Studio in 60 s to save credits (everything is on its persistent disk)."
  sleep 60
  for py in python3 python /opt/conda/bin/python; do
    if command -v "$py" >/dev/null && "$py" -c "from lightning_sdk import Studio; Studio().stop()" 2>/dev/null; then
      return 0
    fi
  done
  echo "Could not stop the Studio automatically - STOP IT NOW in the Lightning UI."
}
stage() { echo; echo "================ [$(date +%H:%M:%S)] $* ================"; }
trap 'echo; echo "RUN FAILED at line $LINENO - see the error above. Fix it and re-run: bash scripts/lightning_gpu_run.sh"; rm -f "$PIDFILE"; echo PROGRESS_EXIT 1; stop_studio' ERR

stage "0/5 environment"
"$UV" sync --frozen --extra train --extra data --group dev
nvidia-smi || true
if ! run_py -c "from finetune.common import Config; from finetune.download_model import verified; import sys; c = Config.load(); sys.exit(0 if verified(c.model_dir, c) else 1)"; then
  echo "WARNING: the base model was not downloaded on CPU first - downloading now (~5-10 min of GPU time)"
  run_py -m finetune.download_model
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # everything needed is on disk now

RUN_DIR="$(run_py -c 'from finetune.common import Config; print(Config.load().output_dir)')"
MODEL="$(run_py -c 'from finetune.common import Config; print(Config.load().raw["base_model"]["repository"])')"
echo "config: ${ACHARYA_CONFIG:-configs/train.yaml} | model: $MODEL | run dir: $RUN_DIR"
echo "GPU budget: ${GPU_MINUTES} min in total; training ends by $(date -d @"$ACHARYA_TRAIN_DEADLINE" +%H:%M) at the latest"

stage "1/5 preflight (GPU, CUDA, bf16, disk, model, dataset, time budget)"
run_py -m finetune.preflight   # stops here with a clear message if anything is wrong

stage "2/5 LoRA fine-tuning of $MODEL (resumes automatically if interrupted)"
echo "Resume points every ~10 min; portable copy: $RUN_DIR/recovery/recovery-latest.tar"
run_py -m finetune.train

stage "3/5 evaluation: base vs fine-tuned on the held-out test set (resumable)"
run_py -m finetune.evaluate

stage "4/5 report + charts"
run_py -m finetune.report

stage "5/5 merge LoRA into a standalone model + small download bundle"
run_py -m finetune.export
# Checkpoints are kept (they are the only way to continue training later); the small
# bundle below is what to download if credits run out.
tar -cf "$RUN_DIR/results-bundle.tar" -C "$RUN_DIR" adapter report eval training_summary.json \
  metrics.jsonl RUN_INFO.json 2>/dev/null || true
GGUF_NOTE="  gguf:     not built yet - switch to CPU, then: bash scripts/export_gguf.sh"

minutes=$(( ($(date +%s) - started) / 60 ))
cat <<EOF

=====================================================================
ALL DONE in ${minutes} min.
  report:   $RUN_DIR/report/REPORT.md  (+ PNG charts)
  adapter:  $RUN_DIR/adapter           (LoRA weights, ~1 GB)
  merged:   $RUN_DIR/merged            (full fine-tuned model)
  bundle:   $RUN_DIR/results-bundle.tar (adapter + report + eval: download this one)
${GGUF_NOTE}

Optional - demo the iOS app against this GPU right now:
    bash scripts/lightning_serve.sh         -> prints an https URL for the app

The H200 keeps billing until you stop the Studio or switch it back to CPU.
Switch to CPU BEFORE exporting the GGUF (it needs no GPU).
=====================================================================
EOF
rm -f "$PIDFILE"
echo PROGRESS_EXIT 0
stop_studio
