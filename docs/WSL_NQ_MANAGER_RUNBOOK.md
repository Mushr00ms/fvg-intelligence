# WSL NQ Bot Manager Runbook

This runbook sets up the existing NQ Interactive Brokers bot manager inside WSL while keeping the live IBKR login session on Windows TWS or IB Gateway.

Current operating model:

- WSL runs the Telegram bot manager and the NQ bot process.
- Windows runs TWS or IB Gateway with the live IBKR session.
- The bot connects from WSL to the Windows-hosted IB API port.
- Telegram remains the control surface for `start`, `stop`, `status`, and `logs`.

This does not automate IBKR login or 2FA. You still need a live Windows TWS or IB Gateway session that is already logged in.

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

## 2. Create an external live config

Do not keep the live config in the repo. Put it under your WSL home directory instead:

```bash
mkdir -p ~/.config/fvg-intelligence
chmod 700 ~/.config/fvg-intelligence
```

Create `~/.config/fvg-intelligence/nq-live-bot.json`:

```json
{
  "ib_host": "172.30.96.1",
  "ib_port": 7496,
  "ib_client_id": 1,
  "paper_mode": false,
  "dry_run": false,
  "telegram_bot_token": "REPLACE_ME",
  "telegram_chat_id": "REPLACE_ME"
}
```

Notes:

- If `ib_host` is omitted or left as `127.0.0.1`, the bot config code will try to auto-detect the Windows host from WSL.
- Keep the file mode tight:

```bash
chmod 600 ~/.config/fvg-intelligence/nq-live-bot.json
```

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
BOT_CONFIG_PATH=/home/<your-user>/.config/fvg-intelligence/nq-live-bot.json
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

## 5. Daily live workflow

1. Start Windows TWS or IB Gateway in live mode.
2. Complete the IBKR login and 2FA on Windows.
3. Confirm the API port is enabled on Windows TWS or Gateway.
4. Let the WSL service start automatically, or start it manually.
5. Use Telegram to control the NQ bot from the panel.

## 6. Failure modes

`Cannot reach IB Gateway/TWS`

- Windows TWS or IB Gateway is not running.
- The live API port is not enabled.
- WSL is using a stale Windows host IP.

`Telegram panel does not update`

- The manager is not running in an environment with outbound internet access.
- The Telegram bot token or chat ID in the external config is wrong.

`Manager starts, bot start fails`

- TWS or Gateway was reachable during preflight but later disconnected.
- Another IB API client is using the same client ID.
- The Windows live session was logged out after startup.

## 7. Security boundary

For this WSL setup, the bot never needs your IBKR username or password.

- IBKR credentials stay on the Windows side where TWS or IB Gateway is logged in.
- WSL only needs the external bot config for IB host/port and Telegram settings.
- Keep Telegram credentials and any future live secrets outside the repo.
