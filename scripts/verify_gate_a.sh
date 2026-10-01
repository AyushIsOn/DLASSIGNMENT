#!/usr/bin/env bash
set -euo pipefail
ROOT="/projects/sandbox/AcharyaGPT/.worktrees/rag-rebuild"
export ACHARYA_WORKSPACE="$ROOT"
export PYENV_VERSION=3.11.15
uv run --project "$ROOT" python -m acharya.gate_a
uv run --project "$ROOT" python -c 'import json, pathlib; p=pathlib.Path("/projects/sandbox/AcharyaGPT/.worktrees/rag-rebuild/artifacts/gates/gate-a.json"); assert json.loads(p.read_text())["status"] == "PASSED"'
