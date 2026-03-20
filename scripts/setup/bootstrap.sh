#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

cd "$REPO_ROOT"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .[dev]
