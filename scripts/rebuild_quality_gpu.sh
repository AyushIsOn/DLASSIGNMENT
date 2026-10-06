#!/usr/bin/env bash
set -euo pipefail
SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "${1:?usage: rebuild_quality_gpu.sh WORKSPACE}" && pwd)"
export PYTHONPATH="$SOURCE/src:$SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false ACHARYA_RETRIEVAL_DEVICE=cuda
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PYTHON="$WORKSPACE/.venv/bin/python"
"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA required')
print('GPU:', torch.cuda.get_device_name(0), flush=True)
PY
OUTPUT="$WORKSPACE/artifacts/quality-rebuild"
mkdir -p "$OUTPUT"
# Preserve a completed comparison; failures leave partial review files intact.
if [[ ! -f "$OUTPUT/direct/manifest.json" ]]; then
  DIRECT="$OUTPUT/direct-$(date -u +%Y%m%dT%H%M%SZ)"
  "$PYTHON" -u -m training.direct_compare --workspace "$WORKSPACE" \
    --adapter "$WORKSPACE/artifacts/external-run/adapter" --output "$DIRECT" --limit 32
  ln -s "$(basename "$DIRECT")" "$OUTPUT/direct"
fi
# Resumable drafting from the base model. Never auto-approve synthetic targets.
"$PYTHON" -u -m training.rewrite_targets --workspace "$WORKSPACE" \
  --output "$OUTPUT/drafts.jsonl" --limit 1500 --minutes 90
"$PYTHON" - "$OUTPUT" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
root = Path(sys.argv[1])
path = root/'drafts.jsonl'
rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
summary = {'drafts': len(rows), 'by_split': dict(Counter(r['split'] for r in rows)),
           'automatically_unflagged': sum(not r['automated_rejection_reasons'] for r in rows),
           'approved_for_training': 0, 'next_step': 'Source-grounded draft review'}
(root/'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)
PY
tar -czf "$OUTPUT.tar.gz" -C "$(dirname "$OUTPUT")" "$(basename "$OUTPUT")"
echo "DOWNLOAD FOR REVIEW: $OUTPUT.tar.gz"
