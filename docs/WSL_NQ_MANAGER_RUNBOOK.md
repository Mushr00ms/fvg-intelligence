# WSL NQ Bot Manager Runbook

This runbook sets up the NQ bot manager inside WSL for the mixed launch mode:
IBKR provides market data, and Tradovate receives demo-account executions.

Current operating model:

- WSL runs the Telegram bot manager and the NQ bot process.
- Windows runs TWS or IB Gateway with the IBKR market-data session.
- The bot connects from WSL to the Windows-hosted IB API port.
- The bot authenticates to Tradovate `demo` for order execution.
- Telegram remains the control surface for `start`, `stop`, `status`, and `logs`.

This does not automate IBKR login or 2FA. You still need a Windows TWS or IB Gateway paper session that is already logged in.

## 1. WSL prerequisites

- WSL can reach the Windows TWS or IB Gateway API port.
- Python dependencies for this repo are installed in WSL.
- Telegram bot token and target chat are known.

Optional but recommended:

- Enable `systemd` in WSL by adding this to `/etc/wsl.conf`:

```ini
[boot]
systemd=true
```

Then restart WSL from Windows:

```powershell
wsl --shutdown
```

## 2. Create an external Tradovate demo config

Do not keep the launch config in the repo. Put it under your WSL home directory instead:

```bash
mkdir -p ~/.config/fvg-intelligence
chmod 700 ~/.config/fvg-intelligence
```

The launcher defaults to the repo-local config at `/mnt/c/Users/cr0wn/fvg-intelligence/bot/bot_config.json`. If you want a separate external config, create `~/.config/fvg-intelligence/nq-tradovate-demo-bot.json`:

```json
{
  "execution_backend": "ib_data_tradovate_exec",
  "tradovate_environment": "demo",
  "tradovate_account_spec": "",
  "ib_host": "172.30.96.1",
  "ib_port": 7497,
  "ib_client_id": 1,
  "paper_mode": true,
  "dry_run": false,
  "telegram_bot_token": "REPLACE_ME",
  "telegram_chat_id": "REPLACE_ME"
}
```

Notes:

- If `ib_host` is omitted or left as `127.0.0.1`, the bot config code will try to auto-detect the Windows host from WSL.
- `dry_run: false` is intentional here: orders are sent to Tradovate demo, not live Tradovate.
- `paper_mode: true` selects the IBKR paper-data socket (`7497`) for market data.
- Set `tradovate_account_spec` if the demo login exposes multiple Tradovate accounts.
- Keep the file mode tight:

```bash
chmod 600 ~/.config/fvg-intelligence/nq-tradovate-demo-bot.json
```

Before launch, run the preflight explicitly:

```bash
cd /mnt/c/Users/cr0wn/fvg-intelligence
bash scripts/run_nq_manager_wsl.sh
```

The launcher runs the preflight before starting the Telegram manager. It fails if the backend is not `ib_data_tradovate_exec`, if Tradovate is not `demo`, if IBKR is not paper mode on `7497`, if the active strategy cannot load, if the IBKR socket is unreachable, or if Tradovate demo credentials cannot be loaded.

Credential notes:

- Under systemd, `LoadCredentialEncrypted` decrypts `~/.config/fvg-intelligence/credentials/*.cred` and exposes plaintext files through `$CREDENTIALS_DIRECTORY`.
- For manual `bash scripts/run_nq_manager_wsl.sh` launches, the launcher decrypts the required Tradovate `.cred` files into a private temporary runtime directory, exports `$CREDENTIALS_DIRECTORY`, and removes the temp directory on exit.

## 3. Start manually from WSL

The repo now includes a WSL launcher that checks config existence and verifies the IB API socket before starting the Telegram manager:

```bash
cd /mnt/c/Users/cr0wn/fvg-intelligence
bash scripts/run_nq_manager_wsl.sh
```

If the Windows TWS or IB Gateway session is not reachable, the launcher exits before the manager starts.

## 4. Install as a WSL systemd service

Create the service environment file from the repo example:

```bash
mkdir -p ~/.config/fvg-intelligence
cp /mnt/c/Users/cr0wn/fvg-intelligence/ops/wsl/nq-bot-manager.env.example \
   ~/.config/fvg-intelligence/nq-bot-manager.env
chmod 600 ~/.config/fvg-intelligence/nq-bot-manager.env
```

Edit `~/.config/fvg-intelligence/nq-bot-manager.env`:

```bash
REPO_DIR=/mnt/c/Users/cr0wn/fvg-intelligence
BOT_CONFIG_PATH=/mnt/c/Users/cr0wn/fvg-intelligence/bot/bot_config.json
PYTHON_BIN=python3
```

Install the service unit:

```bash
mkdir -p ~/.config/systemd/user
cp /mnt/c/Users/cr0wn/fvg-intelligence/ops/wsl/nq-bot-manager.service \
   ~/.config/systemd/user/nq-bot-manager.service
systemctl --user daemon-reload
systemctl --user enable --now nq-bot-manager.service
```

Check status:

```bash
systemctl --user status nq-bot-manager.service
journalctl --user -u nq-bot-manager.service -f
```

## 5. Daily launch workflow

1. Start Windows TWS or IB Gateway for the IBKR market-data session.
2. Complete the IBKR login and 2FA on Windows.
3. Confirm the API port is enabled on Windows TWS or Gateway.
4. Confirm Tradovate demo credentials are available under `~/.config/fvg-intelligence/credentials/`.
5. Run the preflight command above.
6. Let the WSL service start automatically, or start it manually.
7. Use Telegram to control the NQ bot from the panel.

## 6. Failure modes

`Cannot reach IB Gateway/TWS`

- Windows TWS or IB Gateway is not running.
- The paper API port (`7497`) is not enabled.
- WSL is using a stale Windows host IP.

`Telegram panel does not update`

- The manager is not running in an environment with outbound internet access.
- The Telegram bot token or chat ID in the external config is wrong.

`Manager starts, bot start fails`

- TWS or Gateway was reachable during preflight but later disconnected.
- Another IB API client is using the same client ID.
- The Windows paper session was logged out after startup.
- Tradovate demo credentials are missing or the configured `tradovate_account_spec` does not exist.

## 7. Security boundary

For this WSL setup, the bot never needs your IBKR username or password.

- IBKR credentials stay on the Windows side where TWS or IB Gateway is logged in.
- WSL needs the external bot config for IB host/port, Telegram settings, and non-secret Tradovate launch mode.
- Tradovate demo credentials should be stored in SSM or systemd credentials, not in the repo.
- Keep Telegram credentials and Tradovate secrets outside the repo.
