#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "error: activate a virtualenv before running dependency checks" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python -m pip install --require-hashes --only-binary :all: -r "${REPO_ROOT}/requirements.lock"
python -m pip check
