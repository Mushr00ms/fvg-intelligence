#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -eq 0 ]]; then
  echo "Do not run this launcher with sudo." >&2
  echo "Run it from the activated project venv as your normal user:" >&2
  echo "  ./scripts/run_nq_manager_wsl.sh" >&2
  echo "The launcher calls sudo only for systemd-creds decryption when needed." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
CONFIG_HOME="${XDG_CONFIG_HOME:-${HOME}/.config}"
STATE_HOME="${XDG_STATE_HOME:-${HOME}/.local/state}"
REPO_BOT_CONFIG_PATH="${REPO_DIR}/bot/bot_config.json"
EXTERNAL_BOT_CONFIG_PATH="${CONFIG_HOME}/fvg-intelligence/nq-tradovate-demo-bot.json"
DEFAULT_BOT_CONFIG_PATH="${REPO_BOT_CONFIG_PATH}"
BOT_CONFIG_PATH="${BOT_CONFIG_PATH:-${DEFAULT_BOT_CONFIG_PATH}}"
RUNTIME_DIR="${STATE_HOME}/fvg-intelligence"
CRED_DIR="${CONFIG_HOME}/fvg-intelligence/credentials"
LAUNCH_MODE="${LAUNCH_MODE:-bot}"  # bot | manager

mkdir -p "${RUNTIME_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${BOT_CONFIG_PATH}" ]]; then
  requested_path="${BOT_CONFIG_PATH}"
  if [[ "${requested_path}" != "${REPO_BOT_CONFIG_PATH}" && -f "${REPO_BOT_CONFIG_PATH}" ]]; then
    BOT_CONFIG_PATH="${REPO_BOT_CONFIG_PATH}"
    echo "Configured BOT_CONFIG_PATH not found: ${requested_path}" >&2
    echo "Falling back to repo-local config: ${BOT_CONFIG_PATH}" >&2
  elif [[ "${requested_path}" != "${EXTERNAL_BOT_CONFIG_PATH}" && -f "${EXTERNAL_BOT_CONFIG_PATH}" ]]; then
    BOT_CONFIG_PATH="${EXTERNAL_BOT_CONFIG_PATH}"
    echo "Configured BOT_CONFIG_PATH not found: ${requested_path}" >&2
    echo "Falling back to external config: ${BOT_CONFIG_PATH}" >&2
  else
    echo "Bot config not found: ${requested_path}" >&2
    echo "Also checked: ${REPO_BOT_CONFIG_PATH}" >&2
    echo "Also checked: ${EXTERNAL_BOT_CONFIG_PATH}" >&2
    echo "Create the external config file before starting the WSL manager." >&2
    exit 1
  fi
fi

export PYTHONUNBUFFERED=1

# systemd services receive decrypted LoadCredentialEncrypted files through
# $CREDENTIALS_DIRECTORY. For manual launches, decrypt the repo-managed
# encrypted credentials into a private temp dir for this process tree only.
if [[ -z "${CREDENTIALS_DIRECTORY:-}" && -d "${CRED_DIR}" ]]; then
  REQUIRED_CREDS=(
    tradovate-username
    tradovate-password
    tradovate-cid
    tradovate-sec
    tradovate-app_id
    tradovate-device_id
  )
  encrypted_found=false
  for name in "${REQUIRED_CREDS[@]}"; do
    if [[ -f "${CRED_DIR}/${name}.cred" ]]; then
      encrypted_found=true
      break
    fi
  done

  if [[ "${encrypted_found}" == "true" ]]; then
    if ! command -v systemd-creds >/dev/null 2>&1; then
      echo "Encrypted credentials found, but systemd-creds is not available." >&2
      exit 1
    fi

    RUNTIME_CRED_DIR="$(mktemp -d "${RUNTIME_DIR}/credentials.XXXXXX")"
    chmod 0700 "${RUNTIME_CRED_DIR}"
    cleanup_creds() {
      if [[ -n "${RUNTIME_CRED_DIR:-}" && "${RUNTIME_CRED_DIR}" == "${RUNTIME_DIR}/credentials."* ]]; then
        rm -rf "${RUNTIME_CRED_DIR}" 2>/dev/null || sudo rm -rf "${RUNTIME_CRED_DIR}"
      fi
    }
    trap cleanup_creds EXIT

    for name in "${REQUIRED_CREDS[@]}"; do
      src="${CRED_DIR}/${name}.cred"
      dst="${RUNTIME_CRED_DIR}/${name}"
      if [[ ! -f "${src}" ]]; then
        echo "Missing encrypted credential: ${src}" >&2
        exit 1
      fi
      sudo systemd-creds decrypt --name="${name}" "${src}" "${dst}"
      sudo chown "$(id -u):$(id -g)" "${dst}"
      chmod 0600 "${dst}"
    done

    export CREDENTIALS_DIRECTORY="${RUNTIME_CRED_DIR}"
  fi
fi

"${PYTHON_BIN}" "${REPO_DIR}/scripts/preflight_tradovate_demo_launch.py" \
  --config "${BOT_CONFIG_PATH}" \
  --no-dry-run \
  --check-ib-socket \
  --check-secrets

cd "${REPO_DIR}"
if [[ "${LAUNCH_MODE}" == "manager" ]]; then
  echo "Starting Telegram manager. IBKR will connect only after /start or panel Start." >&2
  "${PYTHON_BIN}" -m bot.manager --config "${BOT_CONFIG_PATH}" --no-dry-run
elif [[ "${LAUNCH_MODE}" == "bot" ]]; then
  echo "Starting NQ bot now. IBKR should show an API client connection after startup." >&2
  "${PYTHON_BIN}" -m bot.main --config "${BOT_CONFIG_PATH}" --no-dry-run
else
  echo "Invalid LAUNCH_MODE=${LAUNCH_MODE}. Expected 'bot' or 'manager'." >&2
  exit 1
fi
