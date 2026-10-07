#!/usr/bin/env bash
# OPTIONAL - serve the fine-tuned model from the H200 to the iOS app over HTTPS.
#
#   bash scripts/lightning_serve.sh
#
# Starts the AcharyaGPT API on port 8000 and a free Cloudflare quick tunnel, then prints
# a public https://....trycloudflare.com URL: paste it into the app (gear icon -> Server).
# The URL is random and lives only while this script runs. Ctrl+C stops everything.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p logs artifacts/tools

CF_VERSION="2026.10.0"
CF="$ROOT/artifacts/tools/cloudflared-$CF_VERSION"
if [[ ! -x "$CF" ]]; then
  curl -fsSL -o "$CF" \
    "https://github.com/cloudflare/cloudflared/releases/download/$CF_VERSION/cloudflared-linux-amd64"
  chmod +x "$CF"
fi

MODEL_DIR="${ACHARYA_BASE_MODEL_DIR:-$ROOT/artifacts/models/Qwen3-8B}"
ADAPTER="${ACHARYA_ADAPTER_DIR:-$ROOT/artifacts/run/adapter}"
[[ -f "$ADAPTER/adapter_config.json" ]] || {
  echo "No trained adapter in $ADAPTER - run scripts/lightning_gpu_run.sh first"; exit 1; }

run_py -m acharya.cli serve --backend transformers --model-dir "$MODEL_DIR" \
  --adapter "$ADAPTER" --host 127.0.0.1 --port 8000 > logs/api.log 2>&1 &
API=$!
"$CF" tunnel --no-autoupdate --protocol http2 --url http://127.0.0.1:8000 > logs/tunnel.log 2>&1 &
TUN=$!
trap 'kill $API $TUN 2>/dev/null; echo stopped' EXIT INT TERM

echo "Loading the model on the GPU (about 1 minute)..."
for _ in $(seq 1 180); do
  curl -fs http://127.0.0.1:8000/health/ready >/dev/null 2>&1 && break
  kill -0 $API 2>/dev/null || { echo "API failed to start:"; tail -30 logs/api.log; exit 1; }
  sleep 2
done
URL=""
for _ in $(seq 1 60); do
  URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' logs/tunnel.log | head -1 || true)
  [[ -n "$URL" ]] && break
  sleep 1
done
echo "Local check:"
curl -s -X POST http://127.0.0.1:8000/v1/chat -H 'Content-Type: application/json' \
  -d '{"message":"What is the modern equivalent of Amlapitta?","history":[]}'; echo
echo
echo "============================================================"
echo "  App server URL:  ${URL:-<tunnel failed, see logs/tunnel.log>}"
echo "  (it can take ~1 minute before the URL starts answering)"
echo "  In the app: tap the gear icon -> paste the URL -> Save."
echo "============================================================"
wait $API
