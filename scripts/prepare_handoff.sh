#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=""
FINGERPRINT=""
OUTPUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace) WORKSPACE="$2"; shift 2 ;;
    --fingerprint) FINGERPRINT="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$WORKSPACE" && -n "$FINGERPRINT" && -n "$OUTPUT" ]] || {
  echo "--workspace, --fingerprint, and --output are required" >&2
  exit 2
}
export PYENV_VERSION="${PYENV_VERSION:-3.11.15}"
uv run --project "$WORKSPACE" python -m acharya.bundle create \
  --workspace "$WORKSPACE" --fingerprint "$FINGERPRINT" --output "$OUTPUT"
