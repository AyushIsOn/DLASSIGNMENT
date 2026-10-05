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
  "$PYTHON" -u - "$WORKSPACE" <<'PYCPU'
import json
import sys
from pathlib import Path
from acharya.rag.service import RAGService
from acharya.schemas import ChatRequest

workspace = Path(sys.argv[1])
service = RAGService(workspace)
cases = [json.loads(line) for line in (workspace / "eval/acceptance.jsonl").read_text().splitlines() if line.strip()]
records = []
for case in cases:
    response = service.chat(ChatRequest(message=case["query"], history=[]))
    passed = (
        response.outcome == case["outcome"]
        and all(term.casefold() in response.answer.casefold() for term in case["required_terms"])
        and (case["outcome"] != "answered" or bool(response.citations))
    )
    records.append({"id": case["id"], "passed": passed, "response": response.model_dump(mode="json")})
    print(json.dumps({"case": case["id"], "passed": passed}), flush=True)
output = workspace / "artifacts/external-run/evaluation/retrieval-repair.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(records, indent=2))
if not all(row["passed"] for row in records):
    raise SystemExit(f"CPU retrieval checks failed; keep GPU off. Inspect {output}")
print("CPU retrieval checks passed. GPU generation acceptance is still required.")
PYCPU
elif [[ "$MODE" == gpu ]]; then
  "$PYTHON" -u -m training.acceptance --workspace "$WORKSPACE" \
    --adapter "$WORKSPACE/artifacts/external-run/adapter" \
    --output "$WORKSPACE/artifacts/external-run/evaluation/acceptance-repair.json"
else
  echo 'mode must be cpu or gpu' >&2
  exit 2
fi
