#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$HERE/.deps" "$HERE/.cache" "$HERE/.tmp"
export TMPDIR="$HERE/.tmp" TMP="$HERE/.tmp" TEMP="$HERE/.tmp"
export UV_CACHE_DIR="$HERE/.cache/uv" PIP_CACHE_DIR="$HERE/.cache/pip"
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$(command -v python)" --target "$HERE/.deps" brotli==1.2.0
else
  python -m pip install --target "$HERE/.deps" brotli==1.2.0
fi
