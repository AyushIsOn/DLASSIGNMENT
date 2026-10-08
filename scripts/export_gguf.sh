#!/usr/bin/env bash
# Convert the fine-tuned model to GGUF (Q4_K_M: ~9 GB for 14B, ~5 GB for 8B) for Ollama on a Mac.
# CPU only - run it on the Lightning CPU machine AFTER switching away from the GPU.
#
#   bash scripts/export_gguf.sh            # needs <output_dir>/adapter from configs/train.yaml
#
# Output: artifacts/gguf/acharyagpt-q4_k_m.gguf + artifacts/gguf/Modelfile
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/_env.sh

LLAMA_TAG="b11450"                       # pinned llama.cpp build (Oct 2026)
QUANT="${QUANT:-Q4_K_M}"                  # Q8_0 (~15 GB for 14B) is closer to bf16 if the Mac has RAM
TOOLS="$ROOT/artifacts/tools/llama.cpp-$LLAMA_TAG"
MERGED="$(run_py -c 'from finetune.common import Config; print(Config.load().output_dir / "merged")')"
MODEL_GIB="$(run_py -c 'from finetune.common import Config; c = Config.load(); print(1 + int(sum((c.model_dir / n).stat().st_size for n in c.raw["base_model"]["shards"]) / 2**30))')"
OUT="${ACHARYA_GGUF_DIR:-$ROOT/artifacts/gguf}"
F16="$OUT/acharyagpt-f16.gguf"
FINAL="$OUT/acharyagpt-$(echo "$QUANT" | tr 'A-Z' 'a-z').gguf"
mkdir -p "$OUT" "$ROOT/artifacts/tools"

need_gib() {  # need_gib <GiB> <path>
  local free_kib; free_kib=$(df -Pk "$2" | awk 'NR==2 {print $4}')
  if (( free_kib < $1 * 1024 * 1024 )); then
    echo "ERROR: need ${1} GiB free disk in $2 (have $((free_kib / 1024 / 1024)) GiB)"; exit 1
  fi
}

if [[ -f "$FINAL" ]]; then
  echo "== $FINAL already exists (delete it to rebuild)"
else
  # 1) merged bf16 model (skips itself if already done)
  if [[ ! -f "$MERGED/MERGED_FROM.json" ]]; then need_gib $((MODEL_GIB + 2)) "$ROOT"; fi
  run_py -m finetune.export

  # 2) llama.cpp converter (Python script, same commit as the binaries) + quantize binary
  if [[ ! -x "$TOOLS/bin/llama-quantize" ]]; then
    echo "== downloading llama.cpp $LLAMA_TAG"
    rm -rf "$TOOLS" && mkdir -p "$TOOLS/bin"
    curl -fsSL "https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_TAG/llama-$LLAMA_TAG-bin-ubuntu-x64.tar.gz" \
      | tar -xz -C "$TOOLS/bin" --strip-components=1
    curl -fsSL "https://github.com/ggml-org/llama.cpp/archive/refs/tags/$LLAMA_TAG.tar.gz" \
      | tar -xz -C "$TOOLS" --strip-components=1 \
          "llama.cpp-$LLAMA_TAG/convert_hf_to_gguf.py" "llama.cpp-$LLAMA_TAG/gguf-py" \
          "llama.cpp-$LLAMA_TAG/conversion" "llama.cpp-$LLAMA_TAG/requirements"
  fi
  # llama.cpp's own pinned converter requirements (CPU torch), isolated from the main venv
  if [[ ! -x "$TOOLS/venv/bin/python" ]]; then
    "$UV" venv -q --python 3.11 "$TOOLS/venv"
    (cd "$TOOLS/requirements" && "$UV" pip install -q --python "$TOOLS/venv/bin/python" \
      -r requirements-convert_hf_to_gguf.txt tqdm pyyaml requests)
  fi

  # 3) HF -> GGUF (f16), then quantize
  need_gib $((MODEL_GIB * 4 / 3 + 2)) "$OUT"   # f16 GGUF + quantized copy
  echo "== converting to GGUF (f16)"
  "$TOOLS/venv/bin/python" "$TOOLS/convert_hf_to_gguf.py" "$MERGED" --outtype f16 --outfile "$F16"
  echo "== quantizing to $QUANT"
  LD_LIBRARY_PATH="$TOOLS/bin:${LD_LIBRARY_PATH:-}" "$TOOLS/bin/llama-quantize" "$F16" "$FINAL" "$QUANT"
  rm -f "$F16"
fi

# 4) Ollama Modelfile. The API calls Ollama with raw=true and renders the exact training
#    prompt itself, so the TEMPLATE below only matters for `ollama run` in a terminal.
cat > "$OUT/Modelfile" <<EOF
FROM ./$(basename "$FINAL")
TEMPLATE """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
<think>

</think>

"""
SYSTEM """$(run_py -c 'from acharya.prompting import SYSTEM_PROMPT; print(SYSTEM_PROMPT)')"""
PARAMETER temperature 0
PARAMETER top_k 1
PARAMETER num_ctx 4096
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
EOF
ls -lh "$OUT"
echo
echo "Download artifacts/gguf/ to your Mac, then:"
echo "  cd gguf && ollama create acharyagpt -f Modelfile"
echo "  ollama run acharyagpt \"What is the modern equivalent of Amlapitta?\""
