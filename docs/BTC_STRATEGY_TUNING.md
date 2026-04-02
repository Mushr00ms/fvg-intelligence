# BTC Strategy Tuning

This is the repo-local playbook for changing the live BTC strategy without
guesswork.

## Current live candidate

- Strategy file:
  [btc-5min-wf-train2024-prune2025-ev007-s200-mitonly-p15.json](/abs/path/c:/Users/cr0wn/fvg-intelligence/logic/strategies/btc-5min-wf-train2024-prune2025-ev007-s200-mitonly-p15.json)
- Runtime config path:
  [crypto_bot_config.json](/abs/path/c:/Users/cr0wn/fvg-intelligence/crypto_bot/crypto_bot_config.json)
- Walk-forward logic:
  [walkforward_btc_risk_tiers.py](/abs/path/c:/Users/cr0wn/fvg-intelligence/scripts/walkforward_btc_risk_tiers.py)

Current selection logic:
- train cells on `2020-2024`
- prune on `2025`
- test on `2026 Jan-Feb`
- conservative profile:
  - `min_ev = 0.07`
  - `min_samples = 200`
  - `setups = mit_only`
  - `prune_min_trades = 15`

## Source of truth

Use these files in this order:

1. Audit trades:
   [official_audit_trades.json](/abs/path/c:/Users/cr0wn/fvg-intelligence/scripts/btc_sweep_results/official_audit_trades.json)
   and
   [official_audit_trades_2026.json](/abs/path/c:/Users/cr0wn/fvg-intelligence/scripts/btc_sweep_results/official_audit_trades_2026.json)
2. Combined audit file for walk-forward:
   [official_audit_trades_2020_2026ytd.json](/abs/path/c:/Users/cr0wn/fvg-intelligence/scripts/btc_sweep_results/official_audit_trades_2020_2026ytd.json)
3. Walk-forward result grid:
   [walkforward_btc_risk_tiers_10k_compound_train2024_test2026ytd.json](/abs/path/c:/Users/cr0wn/fvg-intelligence/scripts/btc_sweep_results/walkforward_btc_risk_tiers_10k_compound_train2024_test2026ytd.json)

Do not tune directly from live impressions.

## Tuning loop

1. Refresh audit trades if the history window changed.

`2026` rebuild:
```bash
wsl bash -lc "cd /mnt/c/Users/cr0wn/fvg-intelligence && python3 scripts/btc_official_audit.py --year 2026 --output scripts/btc_sweep_results/official_audit_trades_2026.json"
```

2. Merge the base and new audit files if needed.

3. Run walk-forward on the combined file.

Example:
```bash
python scripts/walkforward_btc_risk_tiers.py ^
  --trades-path scripts/btc_sweep_results/official_audit_trades_2020_2026ytd.json ^
  --train-end 2024 ^
  --validation-year 2025 ^
  --test-year 2026 ^
  --start-balance 10000 ^
  --compound ^
  --output scripts/btc_sweep_results/walkforward_btc_risk_tiers_10k_compound_train2024_test2026ytd.json
```

4. Pick a candidate from the walk-forward file.

Required checks:
- positive validation and test PnL
- acceptable DD in both windows
- enough cells and enough trades/day for live operation
- no obvious dependence on a single high-risk bucket

5. Export the chosen strategy JSON.

Example:
```bash
python scripts/export_btc_walkforward_strategy.py ^
  --trades-path scripts/btc_sweep_results/official_audit_trades_2020_2026ytd.json ^
  --train-end 2024 ^
  --validation-year 2025 ^
  --min-ev 0.07 ^
  --min-samples 200 ^
  --setups mit_only ^
  --prune-min-trades 15 ^
  --id btc-5min-wf-train2024-prune2025-ev007-s200-mitonly-p15
```

6. Run the leverage-aware sim on the exported strategy.

This is mandatory before live use.

Example:
```bash
wsl bash -lc "cd /mnt/c/Users/cr0wn/fvg-intelligence && python3 scripts/sweep_btc_leverage.py --year 2025 --equity 10000 --compound --strategy-path logic/strategies/btc-5min-wf-train2024-prune2025-ev007-s200-mitonly-p15.json"
```

What to check:
- `10x` taken vs missed trades
- peak concurrent positions
- max DD
- liquidation events must stay `0`

7. If the strategy should be visible in dashboards, export it there too.

Root RR dashboard (`/`):
```bash
python scripts/export_btc_strategy_rr_dataset.py ^
  --strategy-path logic/strategies/btc-5min-wf-train2024-prune2025-ev007-s200-mitonly-p15.json ^
  --data-period wf24p25
```

Legacy heatmap workspace (`/heatmap`):
```bash
python scripts/export_btc_strategy_heatmap.py ^
  --strategy-path logic/strategies/btc-5min-wf-train2024-prune2025-ev007-s200-mitonly-p15.json
```

8. Only then wire it into the crypto bot config.

Files to update:
- [config.py](/abs/path/c:/Users/cr0wn/fvg-intelligence/crypto_bot/config.py)
- [crypto_bot_config.example.json](/abs/path/c:/Users/cr0wn/fvg-intelligence/crypto_bot/crypto_bot_config.example.json)
- [crypto_bot_config.json](/abs/path/c:/Users/cr0wn/fvg-intelligence/crypto_bot/crypto_bot_config.json)

## What not to do

- Do not judge a change only on `2025`.
- Do not treat walk-forward edge numbers as live-ready until the leverage sim passes.
- Do not switch the live bot to a candidate that materially increases missed trades at `10x` unless that is intentional.
- Do not reuse a broad `both` setup candidate just because validation PnL is high. That is where overfit showed up.

## Current practical defaults

If no better candidate has been validated, stay with:
- `mit_only`
- `min_ev >= 0.07`
- `min_samples >= 200`
- `prune_min_trades = 15`
- `10x`
- `$10k`
- compounding on

## Fast recap

Use this sequence:
- rebuild audit trades
- rerun walk-forward
- export strategy
- rerun leverage sim
- export to dashboard
- wire into crypto bot
