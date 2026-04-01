# HFOIV Gate Playbook

## Purpose
Add a pre-entry liquidity stress filter inspired by the paper findings on high-frequency order-imbalance volatility (HFOIV).

For large/liquid instruments, short-horizon imbalance volatility (5-minute sampling) can identify inventory-stress regimes where fills/liquidity are worse and spread risk is higher.

## Core Idea
Compute a rolling HFOIV metric and compare it to a recent distribution.

If the current value is in a high percentile (for example above the 80th percentile), either:
- skip entries, or
- cut size (recommended first pass: 50% size).

## Feature Definition
### 1) 5-minute imbalance proxy
At each completed 5-minute bar, compute an imbalance proxy:
- Baseline proxy: `imbalance_t = sign(close_t - open_t) * volume_t`
- Better proxy (if available): aggressor buy/sell delta from tick data.

### 2) Rolling HFOIV
Use a short rolling window:
- `HFOIV_t = std(imbalance_{t-N+1..t})`
- Suggested default: `N = 12` bars (last 60 minutes).

### 3) Intraday normalization
Because intraday liquidity is time-of-day dependent, normalize by bucket:
- Compute percentile rank of current HFOIV vs same intraday bucket history.
- Suggested bucket: 30-minute interval.
- Suggested history: last 60 sessions.

## Gate Policy
Suggested first deployment:
- Threshold: `hfoiv_percentile >= 80`
- Action: `size_cut`
- Multiplier: `0.5`

Alternative stricter policy:
- Action: `skip`

## Suggested Config
```json
"hfoiv_gate": {
  "enabled": true,
  "bar_interval_min": 5,
  "rolling_bars": 12,
  "lookback_sessions": 60,
  "percentile_threshold": 80,
  "action": "size_cut",
  "size_multiplier": 0.5
}
```

## Integration Points
### Backtester
Hook in pre-entry flow, before final setup acceptance/position sizing.

### Bot (live/paper)
Hook in detection-to-order flow, before risk gates/order placement.

## Rollout Plan
1. Implement feature + gate behind config flag (`enabled=false` default).
2. Run A/B tests:
- baseline
- percentile thresholds: 75 / 80 / 85
- actions: `size_cut` vs `skip`
3. Compare:
- net PnL
- profit factor
- max drawdown
- tail-day loss
- trade count
4. Segment results by:
- time bucket
- risk bucket
- setup type

## Notes
- Start with `size_cut` to avoid removing too much positive expectancy.
- If true aggressor flow is not available in all environments, use the close-open-volume proxy consistently across backtest and live.
- Keep this gate strategy-configurable so it can be enabled per strategy variant.
