#!/usr/bin/env bash
# STEP 2 - run on the H200 after scripts/lightning_cpu_setup.sh finished on CPU.
#
#   bash scripts/lightning_gpu_run.sh
#
# preflight -> train (~1 h) -> evaluate base vs fine-tuned (~15 min) -> report -> merge
# -> GGUF for the Mac (best effort). Runs in its own background session: closing the
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
stage() { echo; echo "================ [$(date +%H:%M:%S)] $* ================"; }
trap 'echo; echo "RUN FAILED at line $LINENO - see the error above. Fix it and re-run: bash scripts/lightning_gpu_run.sh"; rm -f "$PIDFILE"; echo PROGRESS_EXIT 1' ERR

stage "0/6 environment"
"$UV" sync --frozen --extra train --extra data --group dev
nvidia-smi || true
if ! run_py -c "from finetune.common import Config; from finetune.download_model import verified; import sys; c = Config.load(); sys.exit(0 if verified(c.model_dir, c) else 1)"; then
  echo "WARNING: the base model was not downloaded on CPU first - downloading now (~5-10 min of GPU time)"
  run_py -m finetune.download_model
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # everything needed is on disk now

stage "1/6 preflight (GPU, CUDA, bf16, disk, model, dataset)"
run_py -m finetune.preflight   # stops here with a clear message if anything is wrong

stage "2/6 LoRA fine-tuning of Qwen3-8B (resumes automatically if interrupted)"
run_py -m finetune.train

stage "3/6 evaluation: base vs fine-tuned on the held-out test set"
run_py -m finetune.evaluate

stage "4/6 report + charts"
run_py -m finetune.report

stage "5/6 merge LoRA into a standalone model"
run_py -m finetune.export
# Training is finished and the best adapter is saved: the optimizer checkpoints (~2 GB each)
# are no longer needed and only cost disk space.
rm -rf artifacts/run/checkpoints

stage "6/6 GGUF for Ollama on a Mac (best effort, ~10 min on CPU cores)"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
if bash scripts/export_gguf.sh; then
  GGUF_NOTE="  gguf:     artifacts/gguf/                (download this folder to your Mac)"
else
  GGUF_NOTE="  gguf:     FAILED (training results are fine) - later, on CPU: bash scripts/export_gguf.sh"
fi

minutes=$(( ($(date +%s) - started) / 60 ))
cat <<EOF

=====================================================================
ALL DONE in ${minutes} min.
  report:   artifacts/run/report/REPORT.md  (+ PNG charts)
  adapter:  artifacts/run/adapter           (LoRA weights, ~700 MB)
  merged:   artifacts/run/merged            (full fine-tuned model)
${GGUF_NOTE}

Optional - demo the iOS app against this GPU right now:
    bash scripts/lightning_serve.sh         -> prints an https URL for the app

The H200 keeps billing until you stop the Studio or switch it back to CPU.
=====================================================================
EOF
rm -f "$PIDFILE"
echo PROGRESS_EXIT 0
