#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# Dependencies must be installed once with all extras, while on the CPU instance.
# --no-sync avoids uv removing optional training libraries during nested commands.
exec uv run --no-sync --project "$ROOT" python -u -m training.run_a100 "$@"
