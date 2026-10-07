#!/usr/bin/env bash
# STEP 1 - run on a (free) CPU Studio BEFORE switching to the H200.
#
#   bash scripts/lightning_cpu_setup.sh
#
# Installs everything, downloads + verifies Qwen3-8B (16.4 GB), checks the dataset with the
# real tokenizer and runs the whole train/eval/report/merge pipeline on a tiny model.
# Everything lands on the Studio's persistent disk, so the GPU session only trains.
# Safe to re-run: finished steps are skipped.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/_env.sh

step() { echo; echo "================ $* ================"; }

step "1/6 Python environment (uv, Python 3.11, locked packages incl. CUDA 12.6 PyTorch)"
"$UV" sync --frozen --extra train --extra data --group dev
run_py -c "import torch, transformers, peft; print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__)"

step "2/6 Disk space"
FREE_GIB=$(df -Pk "$ROOT" | awk 'NR==2 {print int($4 / 1024 / 1024)}')
if [[ -f artifacts/models/Qwen3-8B/.verified.json ]]; then NEED=45; else NEED=62; fi
echo "free: ${FREE_GIB} GiB; needed for model + training + merged model + GGUF: ~${NEED} GiB"
if (( FREE_GIB < 30 )); then echo "ERROR: less than 30 GiB free - free up disk space first"; exit 1; fi
if (( FREE_GIB < NEED )); then
  echo "WARNING: enough for training and evaluation, but maybe not for the merged model + GGUF."
fi

step "3/6 Download + verify Qwen3-8B (resumable)"
run_py -m finetune.download_model

step "4/6 Dataset integrity + prompt/label check with the real tokenizer"
run_py -m finetune.check_data --tokenizer artifacts/models/Qwen3-8B

step "5/6 Unit tests"
run_py -m pytest -q tests

step "6/6 End-to-end pipeline smoke test (tiny model, CPU)"
run_py -m finetune.smoke

echo
echo "CPU SETUP COMPLETE."
echo "Next: in Lightning, switch this Studio's machine to 1x H200, open a terminal and run:"
echo "    cd $ROOT && bash scripts/lightning_gpu_run.sh"
