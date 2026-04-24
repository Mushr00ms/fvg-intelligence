#!/usr/bin/env bash
#
# Provision encrypted systemd credentials for the trading bot.
#
# Credentials are encrypted at rest via systemd-creds (host key) and only
# decrypted by systemd at service start — never stored as plaintext on disk.
# The service uses LoadCredentialEncrypted= to mount them at runtime into
# $CREDENTIALS_DIRECTORY.
#
# Usage:
#   bash ops/wsl/setup_credentials.sh          # interactive prompts
#   bash ops/wsl/setup_credentials.sh --check  # verify encrypted files exist
#   bash ops/wsl/setup_credentials.sh --decrypt tradovate-username  # peek

set -euo pipefail

CRED_DIR="${HOME}/.config/fvg-intelligence/credentials"

TRADOVATE_KEYS=(username password cid sec app_id device_id)
TELEGRAM_KEYS=(bot_token chat_id)

ALL_CREDS=()
for k in "${TRADOVATE_KEYS[@]}"; do ALL_CREDS+=("tradovate-${k}"); done
for k in "${TELEGRAM_KEYS[@]}"; do ALL_CREDS+=("telegram-${k}"); done

check_mode() {
    local missing=0
    for key in "${TRADOVATE_KEYS[@]}"; do
        local f="${CRED_DIR}/tradovate-${key}.cred"
        if [[ -f "$f" ]]; then
            printf "  %-35s OK (encrypted)\n" "tradovate-${key}"
        else
            printf "  %-35s MISSING\n" "tradovate-${key}"
            missing=1
        fi
    done
    for key in "${TELEGRAM_KEYS[@]}"; do
        local f="${CRED_DIR}/telegram-${key}.cred"
        if [[ -f "$f" ]]; then
            printf "  %-35s OK (encrypted)\n" "telegram-${key}"
        else
            printf "  %-35s MISSING (optional)\n" "telegram-${key}"
        fi
    done
    return $missing
}

encrypt_credential() {
    local name="$1" value="$2"
    local out="${CRED_DIR}/${name}.cred"
    printf '%s' "$value" \
        | sudo systemd-creds encrypt --with-key=host --name="${name}" - "$out"
    sudo chown "$(id -u):$(id -g)" "$out"
    chmod 0600 "$out"
}

decrypt_credential() {
    local name="$1"
    local path="${CRED_DIR}/${name}.cred"
    if [[ ! -f "$path" ]]; then
        echo "No encrypted credential found: ${path}" >&2
        return 1
    fi
    sudo systemd-creds decrypt --name="${name}" "$path" -
}

prompt_credential() {
    local name="$1" secret="${2:-false}"
    local path="${CRED_DIR}/${name}.cred"
    local has_existing=false

    if [[ -f "$path" ]]; then
        has_existing=true
    fi

    if [[ "$has_existing" == "true" ]]; then
        printf "  %s [*** encrypted ***]: " "$name"
    else
        printf "  %s: " "$name"
    fi

    local value
    if [[ "$secret" == "true" ]]; then
        read -rs value
        echo
    else
        read -r value
    fi

    if [[ -z "$value" && "$has_existing" == "true" ]]; then
        return
    fi

    if [[ -z "$value" ]]; then
        echo "    skipped (empty)"
        return
    fi

    encrypt_credential "$name" "$value"
    echo "    encrypted + saved"
}

# --check mode
if [[ "${1:-}" == "--check" ]]; then
    echo "Credential status (${CRED_DIR}):"
    check_mode
    exit $?
fi

# --decrypt mode
if [[ "${1:-}" == "--decrypt" ]]; then
    if [[ -z "${2:-}" ]]; then
        echo "Usage: $0 --decrypt <credential-name>" >&2
        exit 1
    fi
    decrypt_credential "$2"
    exit $?
fi

# Preflight: systemd-creds must be available
if ! command -v systemd-creds >/dev/null 2>&1; then
    echo "ERROR: systemd-creds not found. Install systemd (>= 250)." >&2
    exit 1
fi

# Acquire sudo upfront (host key is root-owned)
echo "sudo required to encrypt credentials with the host key."
sudo -v || { echo "ERROR: sudo authentication failed." >&2; exit 1; }

# Verify host key exists
if [[ ! -f /var/lib/systemd/credential.secret ]]; then
    echo "Generating systemd host key for credential encryption..."
    sudo systemd-creds setup
fi

echo "=== FVG Bot Credential Setup (encrypted) ==="
echo "Credentials encrypted with systemd-creds (host key)."
echo "Stored in: ${CRED_DIR}/*.cred"
echo "Press Enter to keep existing value, or type new value."
echo

mkdir -p "$CRED_DIR"
chmod 0700 "$CRED_DIR"

echo "Tradovate:"
prompt_credential "tradovate-username"
prompt_credential "tradovate-password" true
prompt_credential "tradovate-cid"
prompt_credential "tradovate-sec" true
prompt_credential "tradovate-app_id"
prompt_credential "tradovate-device_id"

echo
echo "Telegram (optional, press Enter to skip):"
prompt_credential "telegram-bot_token" true
prompt_credential "telegram-chat_id"

# Clean up any leftover plaintext files from previous setup
for name in "${ALL_CREDS[@]}"; do
    plaintext="${CRED_DIR}/${name}"
    if [[ -f "$plaintext" ]]; then
        rm -f "$plaintext"
        echo "  removed plaintext: ${name}"
    fi
done

echo
echo "Status:"
check_mode || true

echo
echo "Reload the service to pick up changes:"
echo "  systemctl --user daemon-reload"
echo "  systemctl --user restart nq-bot-manager"
