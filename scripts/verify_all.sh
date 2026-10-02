#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ACHARYA_WORKSPACE="$ROOT"

uv sync --project "$ROOT" --frozen --group dev
uv run --project "$ROOT" ruff check "$ROOT/src" "$ROOT/training" "$ROOT/tests"
uv run --project "$ROOT" mypy "$ROOT/src/acharya"
uv run --project "$ROOT" pytest -q -m "not external_network and not gpu and not macos" "$ROOT/tests"
bash "$ROOT/scripts/verify_gate_a.sh"
uv run --project "$ROOT" python "$ROOT/scripts/validate_xcodeproj.py" \
  --project "$ROOT/AcharyaGPT(iOS)/AcharyaGPT(iOS).xcodeproj/project.pbxproj"
uv run --project "$ROOT" python -m acharya.lightning preflight --workspace "$ROOT" --cpu-only-dry-run
# Dataset downloads are explicit. CI exercises the offline MVP without credentials.
if [[ "${ACHARYA_VERIFY_DATA:-0}" == "1" ]]; then
  uv run --project "$ROOT" acharya --workspace "$ROOT" prepare-handoff --strict
  uv run --project "$ROOT" acharya --workspace "$ROOT" verify-preparation --strict
fi
uv run --project "$ROOT" python -m acharya.final_gates --workspace "$ROOT"
