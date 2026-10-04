#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ACHARYA_WORKSPACE="$ROOT"
export ACHARYA_WORKSPACE="$ROOT"
export PYENV_VERSION=3.11.15
uv run --project "$ROOT" python -m acharya.gate_a
uv run --project "$ROOT" python -c 'import json, os, pathlib; p=(pathlib.Path(os.environ["ACHARYA_WORKSPACE"]) / "artifacts/gates/gate-a.json"); assert json.loads(p.read_text())["status"] == "PASSED"'
