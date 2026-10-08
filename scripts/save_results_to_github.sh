#!/usr/bin/env bash
# Save a trained run (adapter + evaluation + report) to GitHub before a Studio goes away.
#
#   export GITHUB_TOKEN=github_pat_...        # token with "Contents: read and write" on the repo
#   bash scripts/save_results_to_github.sh                     # artifacts/run -> branch results-round1
#   bash scripts/save_results_to_github.sh artifacts/run2 round2
#
# Files > 90 MB (the adapter weights) are split into 90 MB parts (GitHub's limit is 100 MB);
# SHA256SUMS lets scripts/restore_results.sh check that everything came back intact.
# The token is only used for this one push; it is not stored in the git config.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUN="${1:-artifacts/run}"
NAME="${2:-round1}"
BRANCH="results-$NAME"
DEST="results/$NAME"
REPO="${GITHUB_REPO:-AyushIsOn/DLASSIGNMENT}"
if [[ -n "${GITHUB_TOKEN:-}" ]]; then URL="https://x-access-token:${GITHUB_TOKEN}@github.com/$REPO.git"
else URL="https://github.com/$REPO.git"; echo "(no GITHUB_TOKEN: using this machine's saved git login)"; fi
[[ -f "$RUN/adapter/adapter_model.safetensors" ]] || { echo "no adapter in $RUN/adapter"; exit 1; }

echo "== collecting $RUN"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/$DEST"
for item in adapter eval report training_summary.json metrics.jsonl RUN_INFO.json; do
  [[ -e "$RUN/$item" ]] && cp -r "$RUN/$item" "$STAGE/$DEST/"
done
(cd "$STAGE/$DEST" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
# split big files
find "$STAGE/$DEST" -type f -size +90M | while read -r big; do
  split -b 90M -d -a 3 "$big" "$big.part-"
  rm "$big"
  echo "split $(basename "$big") into $(ls "$big".part-* | wc -l) parts"
done
du -sh "$STAGE/$DEST"

echo "== committing to branch $BRANCH"
WORK="$STAGE/repo"
git clone -q --depth 1 "$URL" "$WORK"
cd "$WORK"
git checkout -q -B "$BRANCH"
rm -rf "$DEST" && mkdir -p "$(dirname "$DEST")" && cp -r "$STAGE/$DEST" "$DEST"
git add -f "$DEST"
git -c user.name="AcharyaGPT results" -c user.email="results@acharyagpt.local" \
  commit -q -m "Trained $NAME results: adapter, evaluation, report"
echo "== pushing (this can take a few minutes)"
git push -q -f origin "$BRANCH"
echo
echo "SAVED: https://github.com/$REPO/tree/$BRANCH/$DEST"
echo "Restore on another machine:  bash scripts/restore_results.sh $NAME"
