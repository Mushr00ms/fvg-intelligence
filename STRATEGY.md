# FVG Intelligence — NQ 5min Trading Bot Strategy Specification

## Overview

Automated trading bot for NQ E-mini futures executing a backtested Fair Value Gap (FVG) strategy via Interactive Brokers TWS. The strategy is derived from 5-year backtesting across 62,703 mitigated FVGs with 1-minute candle precision for stop/target resolution.

---

## Core Concept

A Fair Value Gap is a 3-candle pattern where price leaves an unfilled gap:
- **Bullish FVG**: third candle's low > first candle's high (gap up)
- **Bearish FVG**: third candle's high < first candle's low (gap down)

The strategy enters when price **mitigates** (returns to touch) the FVG zone, betting on a bounce from the zone with a defined risk/reward.

---

## FVG Detection

**Timeframe**: 5-minute candles for detection

**Detection logic** (3 consecutive 5min candles):
```
Bullish FVG:
  zone_low  = candle1.high
  zone_high = candle3.low
  FVG size  = zone_high - zone_low (must be > 0.25 pts)

Bearish FVG:
  zone_low  = candle3.high
  zone_high = candle1.low
  FVG size  = zone_high - zone_low (must be > 0.25 pts)
```

**Middle candle data** (candle 2) is stored for stop calculation:
- `middle_open`, `middle_low`, `middle_high`

---

## Mitigation Detection

**Timeframe**: 1-minute candles for precision

**Mitigation = price touching the FVG zone** (wick-based):
```
if candle_1min.low <= zone_high AND candle_1min.high >= zone_low:
    FVG is mitigated → trigger trade setup evaluation
```

**Constraints**:
- Search within same trading day only (until 16:00 ET)
- FVG expires if not mitigated by session end
- Only the FIRST mitigation counts

---

## Trade Setups (Extreme Stop Only)

Two entry methods, both using the middle candle extreme as stop:

### MIT+EXTREME (Mitigation Entry)
```
Bullish:
  Entry  = zone_high (limit buy)
  Stop   = middle_low (stop sell)
  Target = entry + (n × risk)

Bearish:
  Entry  = zone_low (limit sell)
  Stop   = middle_high (stop buy)
  Target = entry - (n × risk)
```

### MID+EXTREME (Midpoint Entry)
```
Bullish:
  Entry  = (zone_high + zone_low) / 2 (limit buy)
  Stop   = middle_low (stop sell)
  Target = entry + (n × risk)

Bearish:
  Entry  = (zone_high + zone_low) / 2 (limit sell)
  Stop   = middle_high (stop buy)
  Target = entry - (n × risk)
```

**Risk** = |entry - stop|, rounded to nearest 0.25 (NQ tick size)

**n (R:R ratio)** = varies per cell, from the BEST EV lookup table (1.0R to 3.0R)

---

## Strategy Cell Lookup Table

The bot trades only pre-approved cells from the 5Y backtest. Each cell is defined by:
- **Time period**: 30-minute window (e.g., "10:30-11:00")
- **Risk range**: bucket in points (e.g., "10-15")
- **Setup**: mit_extreme or mid_extreme
- **Best n**: the R:R ratio with highest EV for that cell

**Filter criteria**: 200+ samples over 5 years, positive EV at ALL 9 R:R ratios tested (1.0 to 3.0)

### Target Composition (~4.4 trades/day)

| Risk Bucket | Unique Cells | Setup Entries | Trades/day | Role |
|-------------|-------------|---------------|------------|------|
| 5-10pt | 9 | 14 | 2.37 | Afternoon volume, multi-contract (5 cts @$76k) |
| 10-15pt | 6 | 11 | 1.43 | Morning-lunch core, 3 contracts |
| 15-20pt | 1 | 2 | 0.22 | 10:30 morning momentum, 2 contracts |
| 20-25pt | 2 | 2 | 0.33 | 10:30-11:30 larger swings, 1 contract |
| **TOTAL** | **18** | **29** | **~4.4** | |

18 unique cells produce 29 setup entries because some cells qualify for both MIT+EXT and MID+EXT — the same FVG can trigger both entry types. Coverage spans 09:30 through 15:30 with no dead zones. Some days yield 0-1 trades, others 7-8. Over a month it averages ~93 trades.

### Complete Cell Table

All cells: 200+ samples over 5Y, positive EV at ALL 9 R:R ratios (1.0R to 3.0R), extreme stop only.

| # | Time | Risk | Setup | Best n | WR | EV/R | Med Risk | Samples | ~Trades/day |
|---|------|------|-------|--------|-----|------|----------|---------|-------------|
| 1 | 09:30-10:00 | 10-15pt | MID+EXT | 2.75R | 28.9% | +0.0819 | 12.5pt | 208 | 0.17 |
| 2 | 10:30-11:00 | 5-10pt | MID+EXT | 2.50R | 32.4% | +0.1329 | 7.5pt | 207 | 0.16 |
| 3 | 10:30-11:00 | 10-15pt | MID+EXT | 2.75R | 31.7% | +0.1895 | 12.5pt | 309 | 0.25 |
| 4 | 10:30-11:00 | 10-15pt | MIT+EXT | 3.00R | 31.8% | +0.2736 | 12.25pt | 223 | 0.18 |
| 5 | 10:30-11:00 | 15-20pt | MID+EXT | 2.50R | 31.9% | +0.1158 | 17.0pt | 276 | 0.22 |
| 6 | 10:30-11:00 | 15-20pt | MIT+EXT | 1.75R | 39.8% | +0.0934 | 17.5pt | 254 | 0.20 |
| 7 | 10:30-11:00 | 20-25pt | MIT+EXT | 2.00R | 36.9% | +0.1061 | 22.25pt | 217 | 0.17 |
| 8 | 11:00-11:30 | 5-10pt | MID+EXT | 3.00R | 28.5% | +0.1384 | 8.0pt | 253 | 0.20 |
| 9 | 11:00-11:30 | 10-15pt | MID+EXT | 2.75R | 31.6% | +0.1842 | 12.25pt | 342 | 0.27 |
| 10 | 11:00-11:30 | 10-15pt | MIT+EXT | 2.50R | 33.1% | +0.1585 | 12.38pt | 284 | 0.23 |
| 11 | 11:00-11:30 | 20-25pt | MIT+EXT | 1.00R | 53.5% | +0.0700 | 22.0pt | 200 | 0.16 |
| 12 | 11:30-12:00 | 5-10pt | MID+EXT | 2.50R | 31.6% | +0.1074 | 7.5pt | 335 | 0.27 |
| 13 | 11:30-12:00 | 5-10pt | MIT+EXT | 2.00R | 37.9% | +0.1379 | 7.5pt | 261 | 0.21 |
| 14 | 11:30-12:00 | 10-15pt | MID+EXT | 1.25R | 47.8% | +0.0748 | 12.0pt | 337 | 0.27 |
| 15 | 11:30-12:00 | 10-15pt | MIT+EXT | 2.75R | 30.5% | +0.1430 | 12.25pt | 292 | 0.23 |
| 16 | 12:00-12:30 | 5-10pt | MID+EXT | 3.00R | 27.5% | +0.1008 | 7.5pt | 367 | 0.29 |
| 17 | 12:00-12:30 | 5-10pt | MIT+EXT | 1.25R | 49.6% | +0.1171 | 7.5pt | 282 | 0.22 |
| 18 | 12:30-13:00 | 10-15pt | MID+EXT | 2.25R | 33.8% | +0.0975 | 12.0pt | 302 | 0.24 |
| 19 | 12:30-13:00 | 10-15pt | MIT+EXT | 2.00R | 36.3% | +0.0899 | 12.5pt | 300 | 0.24 |
| 20 | 13:00-13:30 | 5-10pt | MID+EXT | 2.50R | 35.2% | +0.2306 | 7.5pt | 384 | 0.30 |
| 21 | 13:00-13:30 | 5-10pt | MIT+EXT | 2.50R | 34.6% | +0.2121 | 7.5pt | 283 | 0.22 |
| 22 | 13:00-13:30 | 10-15pt | MID+EXT | 2.00R | 38.6% | +0.1583 | 12.0pt | 303 | 0.24 |
| 23 | 13:00-13:30 | 10-15pt | MIT+EXT | 3.00R | 27.2% | +0.0884 | 12.25pt | 305 | 0.24 |
| 24 | 13:30-14:00 | 5-10pt | MID+EXT | 3.00R | 28.9% | +0.1556 | 7.5pt | 443 | 0.35 |
| 25 | 13:30-14:00 | 5-10pt | MIT+EXT | 2.75R | 31.5% | +0.1824 | 7.75pt | 333 | 0.26 |
| 26 | 14:00-14:30 | 5-10pt | MIT+EXT | 2.25R | 35.7% | +0.1606 | 7.5pt | 266 | 0.21 |
| 27 | 14:30-15:00 | 5-10pt | MID+EXT | 2.00R | 39.2% | +0.1772 | 7.25pt | 395 | 0.31 |
| 28 | 14:30-15:00 | 5-10pt | MIT+EXT | 1.50R | 47.7% | +0.1930 | 7.5pt | 329 | 0.26 |
| 29 | 15:00-15:30 | 5-10pt | MID+EXT | 2.00R | 36.0% | +0.0788 | 7.5pt | 342 | 0.27 |

### How to read this table

- **Same FVG, two entries**: Cells #3/#4, #5/#6, #9/#10, etc. share the same time+risk but different setups. When one FVG forms in that window, BOTH setups are evaluated. The MID+EXT entry activates only if price reaches the midpoint.
- **Best n varies widely**: Morning cells (10:30-11:00) favor high R:R (2.5-3.0R, ~32% WR), while lunch/afternoon cells favor lower R:R (1.0-2.0R, ~40-50% WR). This is why BEST mode outperforms a fixed target.
- **Trades/day is per-setup**: When a cell has both MIT+EXT and MID+EXT, the FVG only forms once but can generate up to 2 entries (one at mitigation level, one at midpoint).

### Cell Table JSON Format (for bot consumption)

Loaded from `logic/rr_data/rr_nq_5min_5y_30min.json` and filtered at startup:

```json
{
  "time_period": "10:30-11:00",
  "risk_range": "10-15",
  "setup": "mit_extreme",
  "best_n": 3.0,
  "win_rate": 31.8,
  "ev": 0.2736,
  "median_risk": 12.25,
  "samples": 223
}
```

---

## Position Sizing

- **Account risk**: 1% per trade, recalculated from current account balance
- **NQ point value**: $20 per point
- **Contracts**: floor(risk_budget / (risk_pts × $20))
- **Minimum**: 1 contract

| Risk Bucket | Med Risk | Risk $/ct | Contracts @$76k |
|-------------|----------|-----------|-----------------|
| 5-10pt | 7.5pt | $150 | 5 |
| 10-15pt | 12.2pt | $244 | 3 |
| 15-20pt | 17.5pt | $350 | 2 |
| 20-25pt | 22.0pt | $440 | 1 |

---

## Order Types

All orders use **bracket order** (OCA group) via IB TWS:

1. **Entry**: LIMIT order at calculated entry price
   - Buy limit for bullish, sell limit for bearish
   - Good-til-cancelled within session (auto-cancel at 15:55 ET)
   - If FVG mitigated but entry limit not yet filled → order stays active until session end

2. **Take Profit**: LIMIT order at target price
   - Attached to entry as bracket child
   - Cancels automatically when SL fills (OCA)

3. **Stop Loss**: STOP order at stop price
   - Attached to entry as bracket child
   - Cancels automatically when TP fills (OCA)
   - Expected slippage: ~0.25pt (1 tick) during RTH

---

## Risk Management Hard Gates

### Per-Trade Gates
- Max single trade risk: 1% of current balance
- Max single trade loss: 1.5% of current balance (accounts for slippage)
- Risk in points must fall within an approved cell's risk_range
- All prices rounded to 0.25 (NQ tick)

### Portfolio Gates
- **Max concurrent open positions**: 3 trades at any time
- **Max daily loss**: -3% of starting daily balance → KILL SWITCH
  - Cancel ALL pending orders
  - Flatten ALL open positions at market
  - Halt bot for remainder of session
- **Max daily trades**: 15 (prevent runaway in unusual conditions)

### Time Gates
- No new entries before 09:30 ET or after 15:45 ET
- Mandatory flatten at 15:55 ET (market orders to close everything)
- Cancel all unfilled entry orders at 15:50 ET
- No overnight positions

### Connection Gates
- If TWS disconnects: do NOT place new orders until reconnected
- On reconnect: reconcile all positions and orders before resuming
- If disconnect lasts >5 minutes during market hours: flatten on reconnect

---

## Partial Fill Handling

NQ futures are 1-lot contracts, but the bot may submit multi-contract orders (e.g., 5 contracts for 5-10pt risk cells):

1. **On partial fill of entry**: immediately adjust TP/SL quantities to match filled quantity
2. **If remaining entry quantity not filled within 5 minutes**: cancel unfilled remainder, keep filled portion with adjusted TP/SL
3. **Track per-order**: original qty, filled qty, remaining qty
4. **TP/SL are always synchronized** to the filled entry quantity

---

## Data Flow (Real-Time)

```
IB TWS Real-Time Bars
├── 5min bars → FVG Detection (last 3 bars)
│   └── New FVG? → Add to active_fvgs list
│
├── 1min bars → Mitigation Check (scan all active FVGs)
│   └── FVG mitigated? → Evaluate trade setup
│       ├── Check: is (time_period, risk_range, setup) in approved cells?
│       ├── Check: risk gates pass?
│       ├── Check: max concurrent positions not exceeded?
│       └── YES → Calculate sizing → Place bracket order
│
└── Order events → Position tracking
    ├── Entry filled → Log, track position
    ├── TP filled → Log P&L, remove position
    ├── SL filled → Log P&L, remove position, check daily loss gate
    └── Rejected → Log error, alert
```

### Active FVG Management
- FVGs are added on formation (new 5min bar completes the pattern)
- FVGs are removed when:
  - Mitigated (trade placed or rejected by gates)
  - Session ends (16:00 ET)
  - Invalidated: close price breaches middle candle extreme (optional)

---

## Logging

Structured JSON logging to file + console:

```json
{"ts": "2026-03-22T10:31:00-04:00", "event": "fvg_detected", "type": "bullish", "zone": [19500, 19525], "middle_low": 19490, "middle_high": 19530, "size": 25.0}
{"ts": "2026-03-22T10:35:00-04:00", "event": "mitigation", "fvg_id": "...", "candle_low": 19522, "candle_high": 19540}
{"ts": "2026-03-22T10:35:01-04:00", "event": "order_placed", "setup": "mit_extreme", "side": "BUY", "entry": 19525, "stop": 19490, "target": 19630, "risk_pts": 35, "n": 3.0, "contracts": 2}
{"ts": "2026-03-22T10:35:05-04:00", "event": "order_filled", "order_id": 123, "qty": 2, "avg_price": 19525.0}
{"ts": "2026-03-22T11:02:00-04:00", "event": "tp_filled", "order_id": 124, "qty": 2, "avg_price": 19630.0, "pnl": 4200.0}
{"ts": "2026-03-22T11:02:00-04:00", "event": "daily_pnl", "realized": 4200.0, "open_positions": 1, "daily_pct": "+5.5%"}
```

Log files: `bot/logs/YYYY-MM-DD.jsonl`

---

## State Persistence (Crash Recovery)

Write state to `bot/state/bot_state.json` every 30 seconds and on every state change:

```json
{
  "last_updated": "2026-03-22T10:35:05-04:00",
  "daily_start_balance": 76000,
  "daily_realized_pnl": 0,
  "daily_trade_count": 1,
  "active_fvgs": [...],
  "pending_orders": [...],
  "open_positions": [...],
  "kill_switch_active": false
}
```

On startup:
1. Load state file if today's date matches
2. Reconcile with IB: query open orders + positions
3. Resolve any discrepancies (IB is source of truth)
4. Resume normal operation

---

## Expected Performance (Conservative, 5Y Backtest)

| Metric | Value |
|--------|-------|
| Account | $76,000 |
| Risk per trade | 1% ($760) |
| Trades/day | ~5 |
| Avg net $/trade | ~$82 (after $4.50 commission + 0.25pt stop slippage) |
| Monthly P&L (flat) | ~$8,200 |
| Annual ROI (flat) | ~130% |
| Annual ROI (compounding) | ~210%+ |
| LLN convergence | ~175 trades for 95% confidence |
| Convergence time | ~2 months at 5 trades/day |

---

## Tech Stack

- **Language**: Python 3.11+
- **IB API**: `ib_insync` (async TWS wrapper)
- **Reuse from codebase**:
  - `logic/utils/fvg_detection.py` — `detect_fvg()`
  - `logic/utils/time_utils.py` — `ensure_ny_timezone()`, `create_time_intervals()`
  - `logic/config.py` — `load_market_config()`
  - `logic/rr_data/rr_nq_5min_5y_30min.json` — strategy cell lookup
- **Connection**: TWS port 7497 (paper) / 7496 (live)
- **Contract**: NQ (MNQ for smaller sizing if needed), continuous front month
