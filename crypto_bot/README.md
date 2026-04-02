# Crypto Bot

Standalone Binance USD-M Futures runtime for the BTC FVG strategy, currently configured for `BTCUSDC`.

This codebase is intentionally separate from the `bot/` IB/NQ runtime. It reuses only:
- the BTC strategy artifact in `logic/strategies/`
- generic utilities that are not broker- or asset-specific

## Current scope

- Binance USD-M public market data stream (`1m`, `5m`)
- BTC strategy loading from `btc-5min-wf-train2024-prune2025-ev007-s200-mitonly-p15.json`
- FVG detection on `5m`
- mitigation detection on `1m`
- BTC-native quantity sizing from stop distance, fees, leverage, and symbol filters
- risk gates for:
  - max concurrent positions
  - cumulative open risk
  - daily realized loss halt
  - free-margin checks
  - liquidation-buffer checks
  - margin-usage cap
- dry-run runtime
- live entry + exit placement via Binance REST + user-data stream
- atomic JSON state persistence
- startup reconciliation of bot-managed Binance orders and positions
- reconnect + resync loops for market data and user-data streams

## Important live behavior

- Time basis is `America/New_York`, matching the NQ bot's operator view.
- The runtime now defaults to `BTCUSDC` and derives the correct Binance margin asset from exchange metadata, so balance tracking follows the symbol's settlement asset instead of assuming `USDT`.
- Market-data timestamps, FVG formation/mitigation times, strategy hourly buckets, and daily resets all use New York time.
- Startup history seeding never places historical trades. It only rebuilds active FVG context.
- In live mode the bot refuses startup if Binance already has open positions or orders, unless `allow_start_with_open_positions=true`.
- If `allow_start_with_open_positions=true`, the bot only resumes orders/positions with its own `cb_<group>_<leg>` client IDs. Any unmanaged exposure causes startup failure.
- Partial entry fills are handled conservatively:
  - cancel the remaining entry
  - arm TP/SL for the filled quantity
- `HEDGE` mode is the default and recommended setting for concurrent long/short BTC signals.

## Strategy caveat

- The execution runtime is configured for `BTCUSDC`, but the current strategy artifact was not freshly re-optimized for `BTCUSDC` specifically.
- Treat this as an execution-market switch. Strategy parity still needs separate historical validation if you want quote-asset-specific confidence.
- The repo-local tuning workflow for future strategy changes is documented in [BTC_STRATEGY_TUNING.md](/abs/path/c:/Users/cr0wn/fvg-intelligence/docs/BTC_STRATEGY_TUNING.md).

## Not finished yet

These are the main items still needed before real-money production:

- richer account / funding / liquidation-distance monitoring
- integration tests against Binance testnet or a mocked exchange
- operator dashboards and stronger alert coverage
- live testnet burn-in and exchange-behavior validation before production capital

## Run

1. Copy `crypto_bot/crypto_bot_config.example.json` to `crypto_bot/crypto_bot_config.json`
2. Fill in Binance credentials if you want live mode
3. Dry-run:

```bash
python -m crypto_bot.main
```

4. Live:

```bash
python -m crypto_bot.main --live
```
