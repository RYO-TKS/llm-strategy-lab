#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REPO_TOPLEVEL="$(git -C "$REPO_ROOT" rev-parse --show-toplevel 2>/dev/null || true)"

if [ "$REPO_TOPLEVEL" != "$REPO_ROOT" ]; then
  echo "Initialize llm-strategy-lab as an independent Git repository before tagging."
  exit 1
fi

TAG_NAME="${1:-}"
if [ -z "$TAG_NAME" ]; then
  echo "Usage: $0 v0.1.0"
  exit 1
fi

git -C "$REPO_ROOT" tag -a "$TAG_NAME" -m "Release $TAG_NAME"
echo "Created tag: $TAG_NAME"
