#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_HOME="${XDG_CONFIG_HOME:-${HOME}/.config}"
STATE_HOME="${XDG_STATE_HOME:-${HOME}/.local/state}"
BOT_CONFIG_PATH="${BOT_CONFIG_PATH:-${CONFIG_HOME}/fvg-intelligence/nq-live-bot.json}"
RUNTIME_DIR="${STATE_HOME}/fvg-intelligence"

mkdir -p "${RUNTIME_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${BOT_CONFIG_PATH}" ]]; then
  echo "Bot config not found: ${BOT_CONFIG_PATH}" >&2
  echo "Create the external config file before starting the WSL manager." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1

"${PYTHON_BIN}" - "${REPO_DIR}" "${BOT_CONFIG_PATH}" <<'PY'
import socket
import sys

repo_dir, config_path = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo_dir)

from bot.bot_config import load_bot_config

config = load_bot_config(config_path)
try:
    with socket.create_connection((config.ib_host, config.ib_port), timeout=3):
        pass
except OSError as exc:
    print(
        f"Cannot reach IB Gateway/TWS at {config.ib_host}:{config.ib_port}: {exc}",
        file=sys.stderr,
    )
    sys.exit(1)

print(
    f"WSL preflight OK: IB endpoint {config.ib_host}:{config.ib_port} reachable; "
    f"using config {config_path}",
    file=sys.stderr,
)
PY

cd "${REPO_DIR}"
exec "${PYTHON_BIN}" -m bot.manager --config "${BOT_CONFIG_PATH}" --live --no-dry-run
