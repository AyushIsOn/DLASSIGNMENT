#!/usr/bin/env bash
# Continue an interrupted training run on ANOTHER Studio / Lightning account.
#
#   bash scripts/restore_and_resume.sh /path/to/recovery-latest.tar   # archive you uploaded
#   bash scripts/restore_and_resume.sh hub                            # from ACHARYA_HUB_REPO
#
# Before this, on the new Studio (CPU is fine, no GPU cost):
#   git clone -b kiro/round2-namaste-14b https://github.com/AyushIsOn/DLASSIGNMENT.git acharyagpt
#   cd acharyagpt && bash scripts/lightning_cpu_setup.sh       # env + 14B weights + checks
# then upload the archive (Studio file browser -> Upload) or set ACHARYA_HUB_REPO + HF_TOKEN.
#
# This script unpacks the checkpoint into the run directory and, with --start, also starts
# the GPU run (training continues at the saved step; finished stages are skipped).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/_env.sh

SOURCE="${1:-}"
START="${2:-}"
[[ -n "$SOURCE" ]] || { sed -n '2,15p' "$0"; exit 1; }
RUN_DIR="$(run_py -c 'from finetune.common import Config; print(Config.load().output_dir)')"
mkdir -p "$RUN_DIR"

if [[ "$SOURCE" == "hub" ]]; then
  [[ -n "${ACHARYA_HUB_REPO:-}" ]] || { echo "set ACHARYA_HUB_REPO=<user>/<repo> (and HF_TOKEN)"; exit 1; }
  echo "== downloading recovery archive from huggingface.co/$ACHARYA_HUB_REPO"
  SOURCE="$(run_py -m finetune.recovery download "$ACHARYA_HUB_REPO" "$RUN_DIR/recovery" | tail -1)"
fi
[[ -f "$SOURCE" ]] || { echo "archive not found: $SOURCE"; exit 1; }
tar -tf "$SOURCE" | grep -q '^checkpoints/checkpoint-[0-9]*/trainer_state.json$' \
  || { echo "$SOURCE is not a recovery archive (no checkpoints/checkpoint-*/trainer_state.json)"; exit 1; }

if compgen -G "$RUN_DIR/checkpoints/checkpoint-*" >/dev/null; then
  BACKUP="$RUN_DIR.before-restore-$(date +%Y%m%d_%H%M%S)"
  echo "== $RUN_DIR already has checkpoints; moving them to $BACKUP"
  mv "$RUN_DIR" "$BACKUP" && mkdir -p "$RUN_DIR"
fi
echo "== unpacking $SOURCE -> $RUN_DIR"
tar -xf "$SOURCE" -C "$RUN_DIR"
ls "$RUN_DIR/checkpoints"

# Same code + data + config as the interrupted run? (train.py refuses to resume otherwise)
run_py - "$RUN_DIR" <<'EOF'
import json, sys
from pathlib import Path
from finetune.common import Config
from finetune.data import verify_manifest
from finetune.train import run_fingerprint
config = Config.load()
saved = json.loads((Path(sys.argv[1]) / "RUN_INFO.json").read_text())["fingerprint"]
current = run_fingerprint(config, verify_manifest(config.data_dir))
if saved != current:
    raise SystemExit(f"Fingerprint mismatch ({saved} vs {current}): this checkout's config or "
                     "data differs from the run in the archive. Check out the same branch/commit "
                     "and the same ACHARYA_CONFIG.")
print("fingerprint OK:", current)
EOF

if [[ "$START" == "--start" ]]; then
  exec bash scripts/lightning_gpu_run.sh
fi
echo
echo "Restored. Switch this Studio to the H200, then run:"
echo "    cd $ROOT && ${ACHARYA_CONFIG:+ACHARYA_CONFIG=$ACHARYA_CONFIG }bash scripts/lightning_gpu_run.sh"
