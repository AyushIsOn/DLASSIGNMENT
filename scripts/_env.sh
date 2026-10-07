# Shared shell setup for the Lightning scripts (sourced, not executed).
# Lightning Studios use conda + shell aliases; this finds or installs uv and pins the venv.
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
# Lightning sets UV_LIGHTNING_VIRTUALENV_ROOT to a folder that may not exist; uv then fails.
unset UV_LIGHTNING_VIRTUALENV_ROOT
export UV_PROJECT_ENVIRONMENT="$ROOT/.venv"
export UV_LINK_MODE=copy

if command -v uv >/dev/null 2>&1; then
  UV="$(command -v uv)"
elif [[ -x "$HOME/.local/bin/uv" ]]; then
  UV="$HOME/.local/bin/uv"
else
  echo "== installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
  UV="$HOME/.local/bin/uv"
fi
export UV

run_py() {  # run the project's Python (never the conda base interpreter)
  "$ROOT/.venv/bin/python" -u "$@"
}
