#!/usr/bin/env bash
# Install the fork's Python sources against a fixed upstream CUDA wheel.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${UV_CACHE_DIR:?Set UV_CACHE_DIR to a data disk before installation}"
CONTROL="$(pwd)/../slackkv-control"
test -f "$CONTROL/deployment/vllm-installed.txt"
uv venv --python 3.12 .venv
export VLLM_USE_PRECOMPILED=1
# Last upstream commit before the SlackKV-only Python changes; never use latest.
export VLLM_PRECOMPILED_WHEEL_COMMIT=2cf0a6915ce544dc493a0990f2ea38d81601128a
export VLLM_PRECOMPILED_WHEEL_VARIANT=cu129
uv pip install --python .venv/bin/python --torch-backend=cu129 \
  --constraint "$CONTROL/deployment/vllm-installed.txt" -e .
echo 'Installed editable fork. Run the GPU/import checks in MIGRATION.md next.'
