#!/usr/bin/env bash
set -euo pipefail
SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "${1:?usage: final_gpu_report.sh WORKSPACE}" && pwd)"
export PYTHONPATH="$SOURCE/src:$SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export ACHARYA_RETRIEVAL_DEVICE=cuda PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PYTHON="$WORKSPACE/.venv/bin/python"
ADAPTER="$WORKSPACE/artifacts/external-run/adapter"
test -f "$ADAPTER/adapter_model.safetensors"
"$PYTHON" - <<'PY'
import torch
import matplotlib
if not torch.cuda.is_available():
    raise SystemExit('CUDA is required')
print('GPU:', torch.cuda.get_device_name(0), flush=True)
PY
OUTPUT="$WORKSPACE/artifacts/final-report/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$(dirname "$OUTPUT")"
# Existing index/calibration are used unchanged. No training or recalibration.
"$PYTHON" -u -m training.quality_eval --workspace "$WORKSPACE" \
  --adapter "$ADAPTER" --benchmark "$SOURCE/eval/independent_quality.jsonl" \
  --mode gpu --output "$OUTPUT"
"$PYTHON" -m training.final_report --workspace "$WORKSPACE" --results "$OUTPUT"
tar -czf "$OUTPUT.tar.gz" -C "$(dirname "$OUTPUT")" "$(basename "$OUTPUT")"
echo "REPORT AND GRAPHS: $OUTPUT"
echo "DOWNLOAD: $OUTPUT.tar.gz"
