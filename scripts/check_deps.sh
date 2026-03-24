#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "error: activate a virtualenv before running dependency checks" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEELHOUSE="${REPO_ROOT}/vendor/wheels"

if [[ ! -d "${WHEELHOUSE}" ]] || ! compgen -G "${WHEELHOUSE}/*" >/dev/null; then
  echo "error: local wheel cache missing; run ./scripts/build_wheelhouse.sh first" >&2
  exit 1
fi

python -m pip install \
  --require-hashes \
  --only-binary :all: \
  --no-index \
  --find-links "${WHEELHOUSE}" \
  -r "${REPO_ROOT}/requirements.lock"
python -m pip check
