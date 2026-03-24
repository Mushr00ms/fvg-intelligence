#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEELHOUSE="${REPO_ROOT}/vendor/wheels"

mkdir -p "${WHEELHOUSE}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/lib_tools_venv.sh"
activate_tools_venv

python -m pip download \
  --require-hashes \
  --only-binary :all: \
  -r "${REPO_ROOT}/requirements.lock" \
  -d "${WHEELHOUSE}"

echo "Wheel cache refreshed in ${WHEELHOUSE}"
