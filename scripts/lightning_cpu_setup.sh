#!/usr/bin/env bash
# STEP 1 - run on a (free) CPU Studio BEFORE switching to the H200.
#
#   bash scripts/lightning_cpu_setup.sh
#
# Installs everything, downloads + verifies the base model (Qwen3-14B, 29.5 GB), checks the
# dataset with the real tokenizer, builds the SQLite knowledge database and runs the whole
# train/eval/report/merge pipeline on a tiny model. Everything lands on the Studio's
# persistent disk, so the GPU session only trains. Safe to re-run: finished steps are skipped.
# 8B instead of 14B:  ACHARYA_CONFIG=configs/train_8b.yaml bash scripts/lightning_cpu_setup.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/_env.sh

step() { echo; echo "================ $* ================"; }

step "1/7 Python environment (uv, Python 3.11, locked packages incl. CUDA 12.6 PyTorch)"
"$UV" sync --frozen --extra train --extra data --group dev
run_py -c "import torch, transformers, peft; print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__)"
MODEL_DIR="$(run_py -c 'from finetune.common import Config; print(Config.load().model_dir)')"
echo "config: ${ACHARYA_CONFIG:-configs/train.yaml} | base model: $MODEL_DIR"

step "2/7 Disk space"
FREE_GIB=$(df -Pk "$ROOT" | awk 'NR==2 {print int($4 / 1024 / 1024)}')
SIZE_GIB=$(run_py -c 'from finetune.common import Config; import sys; print(30 if "14B" in str(Config.load().model_dir) else 17)')
if [[ -f "$MODEL_DIR/.verified.json" ]]; then MODEL_NEED=0; else MODEL_NEED=$SIZE_GIB; fi
NEED=$(( MODEL_NEED + SIZE_GIB + 25 ))   # model + merged model + checkpoints/recovery/eval
echo "free: ${FREE_GIB} GiB; needed: ~${NEED} GiB (GGUF export later needs ~$((SIZE_GIB + 10)) GiB more)"
if (( FREE_GIB < MODEL_NEED + 25 )); then
  echo "ERROR: not enough disk. Free space with e.g.:"
  echo "   rm -rf artifacts/run/checkpoints artifacts/run/merged artifacts/gguf/*.gguf   # round-1 leftovers"
  exit 1
fi
if (( FREE_GIB < NEED )); then
  echo "WARNING: enough for training and evaluation, but maybe not for the merged model."
fi

step "3/7 Download + verify the base model (resumable)"
run_py -m finetune.download_model

step "4/7 Dataset integrity + prompt/label check with the real tokenizer"
run_py -m finetune.check_data --tokenizer "$MODEL_DIR"

step "5/7 SQLite knowledge database (data/db/acharya.sqlite) + rebuild determinism check"
mkdir -p logs
if run_py -m finetune.build_dataset --output artifacts/rebuild/sft --kb-output artifacts/rebuild/kb \
     > logs/build_dataset.log 2>&1; then
  run_py - <<'EOF'
import json
from pathlib import Path
committed = json.loads(Path("data/sft/MANIFEST.json").read_text())["outputs"]
rebuilt = json.loads(Path("artifacts/rebuild/sft/MANIFEST.json").read_text())["outputs"]
same = committed == rebuilt
print("rebuild identical to the committed dataset:", same)
if not same:
    raise SystemExit("WARNING: the rebuilt dataset differs from data/sft (training still uses "
                     "the committed files, which MANIFEST.json verifies)")
EOF
  echo "database: data/db/acharya.sqlite  (sqlite3 data/db/acharya.sqlite '.tables')"
else
  echo "WARNING: database build skipped (Kaggle download failed?) - see logs/build_dataset.log."
  echo "         Training does not need it: it uses the committed data/sft files."
fi
rm -rf artifacts/rebuild

step "6/7 Unit tests"
run_py -m pytest -q tests

step "7/7 End-to-end pipeline smoke test (tiny model, CPU)"
run_py -m finetune.smoke

echo
echo "CPU SETUP COMPLETE."
echo "Next: in Lightning, switch this Studio's machine to 1x H200, open a terminal and run:"
echo "    cd $ROOT && ${ACHARYA_CONFIG:+ACHARYA_CONFIG=$ACHARYA_CONFIG }bash scripts/lightning_gpu_run.sh"
