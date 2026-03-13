#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec pixi run pyrefly lsp "$@"
