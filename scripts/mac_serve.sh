#!/usr/bin/env bash
# Run AcharyaGPT on your Mac for the iOS Simulator (no GPU and no PyTorch needed).
#
#   1. Install Ollama: https://ollama.com/download  (and start it)
#   2. Copy the exported folder from Lightning to  <repo>/artifacts/gguf/
#      (acharyagpt-q4_k_m.gguf + Modelfile, made by scripts/export_gguf.sh)
#   3. bash scripts/mac_serve.sh            # simulator: http://127.0.0.1:8000
#      bash scripts/mac_serve.sh --lan      # real iPhone on the same Wi-Fi
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/_env.sh
HOST=127.0.0.1
[[ "${1:-}" == "--lan" ]] && HOST=0.0.0.0

command -v ollama >/dev/null || { echo "Install Ollama first: https://ollama.com/download"; exit 1; }
if ! curl -fs http://127.0.0.1:11434/api/version >/dev/null; then
  echo "Starting Ollama..."; (ollama serve > /tmp/ollama.log 2>&1 &); sleep 3
fi
if ! ollama show acharyagpt >/dev/null 2>&1; then
  [[ -f artifacts/gguf/Modelfile ]] || { echo "Missing artifacts/gguf/Modelfile + .gguf"; exit 1; }
  (cd artifacts/gguf && ollama create acharyagpt -f Modelfile)
fi
"$UV" sync --frozen --no-dev   # only fastapi/uvicorn/pydantic/pyyaml - no torch
if [[ "$HOST" == "0.0.0.0" ]]; then
  IP=$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}')
  echo "On the iPhone, set the server URL to: http://$IP:8000"
fi
exec "$ROOT/.venv/bin/python" -m acharya.cli serve --backend ollama --model acharyagpt \
  --host "$HOST" --port 8000
