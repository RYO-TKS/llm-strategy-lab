#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

if [ ! -x "$REPO_ROOT/.venv/bin/python" ]; then
  echo "Virtualenv not found. Run 'make setup' first."
  exit 1
fi

cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT/packages/core/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$REPO_ROOT/.venv/bin/python" -m llm_strategy_lab.cli \
  --config "$REPO_ROOT/configs/experiments/sample_research.yaml"
