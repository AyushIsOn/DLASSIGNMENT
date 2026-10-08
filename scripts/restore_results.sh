#!/usr/bin/env bash
# Get a saved run back from GitHub (made by scripts/save_results_to_github.sh).
#
#   bash scripts/restore_results.sh                 # branch results-round1 -> artifacts/run
#   bash scripts/restore_results.sh round2 artifacts/run2
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
NAME="${1:-round1}"
OUT="${2:-artifacts/run}"
BRANCH="results-$NAME"

echo "== downloading branch $BRANCH"
git fetch -q origin "$BRANCH"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
git --work-tree="$TMP" checkout "origin/$BRANCH" -- "results/$NAME"
git reset -q -- "results/$NAME" 2>/dev/null || true
SRC="$TMP/results/$NAME"

echo "== joining split files"
find "$SRC" -name '*.part-000' | while read -r first; do
  whole="${first%.part-000}"
  cat "$whole".part-* > "$whole"
  rm "$whole".part-*
done
(cd "$SRC" && sha256sum -c --quiet SHA256SUMS) || { echo "CHECKSUM MISMATCH - try again"; exit 1; }

mkdir -p "$OUT"
cp -r "$SRC"/. "$OUT"/
rm -f "$OUT/SHA256SUMS"
echo "RESTORED to $OUT (checksums OK):"
ls "$OUT" "$OUT/adapter"
