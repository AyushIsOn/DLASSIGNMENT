#!/usr/bin/env bash
set -euo pipefail
SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$SOURCE/scripts/quality_cpu.sh" "${1:?usage: quality_gpu.sh WORKSPACE}" cuda
