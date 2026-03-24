#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SECURITY_DIR="${REPO_ROOT}/security"

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/lib_tools_venv.sh"
activate_tools_venv

if ! python -m pip show pip-audit >/dev/null 2>&1 || \
   ! python -m pip show cyclonedx-bom >/dev/null 2>&1; then
  python -m pip install -U pip-audit cyclonedx-bom >/dev/null
fi

pip-audit -r "${REPO_ROOT}/requirements.lock"

mkdir -p "${SECURITY_DIR}"
CYCLO="${TOOLS_VENV}/bin/cyclonedx-py"
if [[ ! -x "${CYCLO}" ]]; then
  echo "error: cyclonedx-py CLI missing in ${TOOLS_VENV}" >&2
  exit 1
fi
"${CYCLO}" requirements \
  "${REPO_ROOT}/requirements.lock" \
  --of JSON \
  -o "${SECURITY_DIR}/sbom.json" \
  --validate

echo "SBOM written to ${SECURITY_DIR}/sbom.json"
