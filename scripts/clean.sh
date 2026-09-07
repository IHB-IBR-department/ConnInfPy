#!/usr/bin/env bash
# Remove regenerable local artifacts: build outputs and tool caches.
# Safe by construction — everything here is gitignored.
set -euo pipefail
cd "$(dirname "$0")/.."

rm -rf build/ dist/ conninfpy.egg-info/
rm -rf .pytest_cache/ .mypy_cache/
find . -type d -name "__pycache__" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true

echo "cleaned: build/ dist/ egg-info/ .pytest_cache/ .mypy_cache/ __pycache__/"
