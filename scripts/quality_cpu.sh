#!/usr/bin/env bash
set -euo pipefail
SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "${1:?usage: quality_cpu.sh WORKSPACE}" && pwd)"
export PYTHONPATH="$SOURCE/src:$SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$SOURCE"
OUTPUT="$WORKSPACE/artifacts/quality/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUTPUT"
DATA="$("$WORKSPACE/.venv/bin/python" - "$WORKSPACE" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
state = json.loads((root / 'artifacts/state/active_preparation.json').read_text())
data = Path(state['processed_path'])
print(data if data.is_absolute() else root / data)
PY
)"
"$WORKSPACE/.venv/bin/python" -u -m training.quality_data audit \
  --data "$DATA" --output "$OUTPUT/old-target-audit.json"
"$WORKSPACE/.venv/bin/python" -u -m training.quality_eval \
  --workspace "$WORKSPACE" --benchmark "$SOURCE/eval/independent_quality.jsonl" \
  --mode cpu --output "$OUTPUT/independent-extractive"
echo "Quality audit saved to $OUTPUT. No GPU training was started."
