#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HERE/.deps${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" "$HERE/compress.py" "$@"
