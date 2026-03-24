#!/usr/bin/env bash
# Helper to bootstrap and activate the shared tooling virtualenv.

# shellcheck disable=SC2034
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_VENV="${TOOLS_VENV:-${REPO_ROOT}/.tools-venv}"

activate_tools_venv() {
  if [[ ! -x "${TOOLS_VENV}/bin/python" ]]; then
    python3 -m venv "${TOOLS_VENV}"
    "${TOOLS_VENV}/bin/python" -m pip install -U pip >/dev/null
  fi
  # shellcheck disable=SC1090
  source "${TOOLS_VENV}/bin/activate"
}
