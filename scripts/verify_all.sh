#!/usr/bin/env bash
set -euo pipefail
ROOT="/projects/sandbox/AcharyaGPT/.worktrees/rag-rebuild"
export PYENV_VERSION="${PYENV_VERSION:-3.11.15}"

uv lock --project "$ROOT" --check
uv sync --project "$ROOT" --frozen --extra retrieval --extra training --group dev
uv run --project "$ROOT" ruff check "$ROOT/src" "$ROOT/training" "$ROOT/tests"
uv run --project "$ROOT" mypy "$ROOT/src/acharya"
uv run --project "$ROOT" pytest -q -m "not external_network and not gpu and not macos" "$ROOT/tests"
bash "$ROOT/scripts/verify_gate_a.sh"
uv run --project "$ROOT" python "$ROOT/scripts/validate_xcodeproj.py" \
  --project "$ROOT/AcharyaGPT(iOS)/AcharyaGPT(iOS).xcodeproj/project.pbxproj"
uv run --project "$ROOT" python -m acharya.lightning preflight \
  --workspace "$ROOT" --cpu-only-dry-run

PREPARED=0
if uv run --project "$ROOT" python -m acharya.cli prepare-handoff \
  --workspace "$ROOT" --strict; then
  uv run --project "$ROOT" python -m acharya.cli prepare-handoff \
    --workspace "$ROOT" --strict
  uv run --project "$ROOT" python -m acharya.cli verify-preparation \
    --workspace "$ROOT" --strict
  PREPARED=1
else
  uv run --project "$ROOT" python -c \
    'import json, pathlib; p=pathlib.Path("/projects/sandbox/AcharyaGPT/.worktrees/rag-rebuild/artifacts/gates/gate-b.json"); assert json.loads(p.read_text())["status"] == "BLOCKED_EXTERNAL_DATA"'
fi

# The provider command alone may read .env.local; its nonzero external status is evidence, not a test failure.
uv run --project "$ROOT" python -m acharya.cli provider-check --workspace "$ROOT" \
  --provider kiro-cli --env-file "$ROOT/.env.local" \
  --record "$ROOT/artifacts/provider-check/kiro.json" || [[ $? -eq 2 ]]

if (( PREPARED == 1 )); then
  uv run --project "$ROOT" python -m acharya.cli index build --workspace "$ROOT" \
    --retrieval-mode full
  uv run --project "$ROOT" python -m acharya.cli evaluate --workspace "$ROOT" \
    --retrieval-mode full --golden "$ROOT/eval/golden.jsonl" --activate-calibration
  FINGERPRINT="$(uv run --project "$ROOT" python -c \
    'import json, pathlib; print(json.loads((pathlib.Path("/projects/sandbox/AcharyaGPT/.worktrees/rag-rebuild/artifacts/state/active_preparation.json")).read_text())["fingerprint"])')"
  mkdir -p "$ROOT/dist/lightning"
  FIRST="$ROOT/dist/lightning/acharyagpt-lightning-${FINGERPRINT}-one.tar.zst"
  SECOND="$ROOT/dist/lightning/acharyagpt-lightning-${FINGERPRINT}-two.tar.zst"
  bash "$ROOT/scripts/prepare_handoff.sh" --workspace "$ROOT" \
    --fingerprint "$FINGERPRINT" --output "$FIRST"
  bash "$ROOT/scripts/prepare_handoff.sh" --workspace "$ROOT" \
    --fingerprint "$FINGERPRINT" --output "$SECOND"
  [[ "$(sha256sum "$FIRST" | cut -d' ' -f1)" == "$(sha256sum "$SECOND" | cut -d' ' -f1)" ]]
  uv run --project "$ROOT" python -m acharya.bundle validate-archive --archive "$FIRST"
  uv run --project "$ROOT" python -m acharya.bundle validate-archive --archive "$SECOND"
fi

uv run --project "$ROOT" python -m acharya.final_gates --workspace "$ROOT"
