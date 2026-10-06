#!/usr/bin/env bash
set -euo pipefail
SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "${1:?usage: quality_cpu.sh WORKSPACE}" && pwd)"
export PYTHONPATH="$SOURCE/src:$SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$SOURCE"
(while sleep 30; do echo "CPU quality workflow still running ($(date -u +%H:%M:%S))."; done) &
HEARTBEAT_PID=$!
trap 'kill "$HEARTBEAT_PID" 2>/dev/null || true' EXIT
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
# Reuse only a validated index with the current passage representation.
if "$WORKSPACE/.venv/bin/python" - "$WORKSPACE" <<'PYINDEX'
import sys
from pathlib import Path
from acharya.config import Settings, sha256_bytes
from acharya.rag.index import load_index
try:
    index = load_index(Settings.load(Path(sys.argv[1])), expected_mode="full")
    if (index.model_hashes or {}).get("passage_format") != sha256_bytes(b"source-question-and-text-v1"):
        raise RuntimeError("old passage representation")
except (OSError, ValueError, KeyError, RuntimeError) as exc:
    print(f"Index rebuild required: {exc}")
    sys.exit(1)
print("Reusing validated full retrieval index.")
PYINDEX
then
  :
else
  "$WORKSPACE/.venv/bin/python" -u -m acharya.cli index build \
    --workspace "$WORKSPACE" --retrieval-mode full
fi
"$WORKSPACE/.venv/bin/python" -u -m acharya.cli evaluate \
  --workspace "$WORKSPACE" --retrieval-mode full \
  --golden "$SOURCE/eval/retrieval_development.jsonl" --activate-calibration
"$WORKSPACE/.venv/bin/python" -u -m training.quality_eval \
  --workspace "$WORKSPACE" --benchmark "$SOURCE/eval/independent_quality.jsonl" \
  --mode cpu --output "$OUTPUT/independent-extractive"
echo "Quality audit saved to $OUTPUT. No GPU training was started."
