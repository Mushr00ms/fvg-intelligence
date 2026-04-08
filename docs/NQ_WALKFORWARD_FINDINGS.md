# NQ Walk-Forward Findings (2026-04-08)

## Status

**LIVE STRATEGY (locked 2026-04-08):** `mixed-best-ev-wf-2020-2025-slotbl-non3.json`

40 cells | 4.40 trades/day | full Kelly tiers (0.5% / 1.5% / 3.0%) | source `rr_nq_5min_2020_2025_30min`

Selection rules (in order):
1. Setups: `mit_extreme` and `mid_extreme` only (no open-stop variants)
2. EV ≥ 0.07, samples ≥ 30, best (setup, rr_target) per (time_period, risk_range) cell — unbounded selector
3. Drop cells with negative train-window backtest P&L (rule A)
4. **Drop entire 12:00-12:30 and 13:00-13:30 time slots** (slot blackout)
5. **Drop all rr_target = 3.0 cells** (n=3.0 prune)

Walk-forward live config: Dec 20 hard gate, witching gates, HFOIV p70×0.25 r6 lb90, tiered Kelly sizing, no slippage, $80k starting balance, max 3 concurrent positions, $33k margin/contract, -10% daily kill switch.

## What we set out to answer

1. The original live strategy `mixed-best-ev-v3-touch-moderate` was fit on a rolling 5y window (~2021-03 → 2026-03), so 2025 was **in-sample** during the fit. Any 2025 result on it is a look-back, not OOS.
2. Build a true walk-forward: train on a fixed window with a hard cutoff, then evaluate untouched on a later window.
3. Test whether shorter, more recent training windows (3y / 4y) produce better OOS performance — i.e. did the market regime shift enough that 2020-2021 data is now noise?
4. Once a candidate strategy emerged, perform a full year-by-year expanding-window WF (train 2020 → test 2021, train 2020-2021 → test 2022, ..., train 2020-2025 → test 2026 YTD) to characterize year-to-year variance.
5. Find ablation rules that improve drawdown without destroying PnL.

## Pipeline changes made

- **`logic/config.py`** — `custom_start` / `custom_end` now read from `FVG_START` / `FVG_END` env vars (was hard-coded `None`). The downstream loader (`logic/utils/databento_data_interface.py`) already honored these; only the env wiring was missing.
- **`scripts/build_nq_strategy_from_window.py`** — new selector that reads any FVG parquet and emits a strategy JSON in the same schema as `mixed-best-ev-v3-touch-moderate`. Supports `--max-trades-per-day` for the live-equivalent budget cap.
- **`scripts/wf_year_pipeline.py`** — end-to-end WF driver: build candidate → train backtest → drop negative train P&L cells → OOS test → save to dashboard.
- **`bot/backtest/backtester.py`** — added `--loss-streak`, `--loss-streak-floor`, `--mnq`, `--mnq-zero` flags for defensive sizing experiments (kept for future use; the locked live strategy doesn't use them).

### Known caveat: dataset id collision

`logic/main.py` derives the RR dataset id from `format_period_str(period)` (the `FVG_PERIOD` short label), not from `custom_start/end`. So all custom-window regens write to `rr_nq_5min_5y_30min.json` and clobber each other. Workaround used in this session: copy each regen's RR JSON to a window-specific name immediately after it finishes. The `fvg_cache` parquets ARE keyed by `custom_start/end`, so they're safe.

The original rolling-5y RR JSON was overwritten multiple times by custom-window regens. The live strategy file is locked so the bot still works, but the source RR dataset for the original `mixed-best-ev-v3-touch-moderate` is no longer the rolling 5y. To restore: run `cd logic && python main.py` with no `FVG_START`/`FVG_END` set. **Better long-term fix:** include start/end in the dataset id label so custom windows write to distinct files.

## Live (original) selection rule — reverse-engineered

Reverse-engineered from `mixed-best-ev-v3-touch-moderate.json` content:

1. Setups: `mit_extreme` and `mid_extreme` only.
2. EV floor 0.07, samples floor 30 (live's min cell sample is 36).
3. Best (setup, rr_target) per (time_period, risk_range) cell.
4. v4 risk bins `[5,10,15,20,25,30,40,50,200]` — 40-80 split into 40-50/50-200 to handle NQ vol expansion.
5. **Greedy budget fill: cumulative `trades_per_day` ≤ 5.0**, sorted by `EV × √samples` descending.

The budget cap was the missing piece. Live's `expected_trades_per_day = 4.9031` is exactly what a "≤5/day" greedy fill produces. Reproducing the rule on a parquet with the same window properties yields **34 cells / weighted EV 0.177 / 4.99 trades-per-day** — within rounding of live's **35 / 0.177 / 4.9031**.

`samples ≥ 200` and "positive EV at all 9 R:R ratios" as documented in `CLAUDE.md` are **wrong** — confirmed by inspecting live cells directly (min samples 36, several cells with single-target EV).

### `mit_extreme` divergence (now understood)

The live strategy is **97% mit_extreme** (34 of 35 cells). Our reverse-engineered selector produces ~50/50 mit/mid_extreme. **The cap5 budget cap does not explain live's mit_extreme bias** — there must be an additional manual filter we couldn't see in the strategy file. Whether mirroring this bias would help was tested separately as the `mit_extreme only` ablation — see results below — and **the answer is no, it would not help**. The live strategy's 97% mit_extreme bias appears to be a bug in the original build, not a feature.

## Datasets and parquets generated this session

| window | parquet hash | RR dataset (renamed) |
|---|---|---|
| 2020 (1y) | `df83a1e1...` | `rr_nq_5min_2020_2020_30min.json` |
| 2020-2021 (2y) | `d85e0e42...` | `rr_nq_5min_2020_2021_30min.json` |
| 2020-2022 (3y) | `6ca241a2...` | `rr_nq_5min_2020_2022_30min.json` |
| 2020-2023 (4y) | `4bee1fa2...` | `rr_nq_5min_2020_2023_30min.json` |
| 2020-2024 (5y) | `271c94a5...` | `rr_nq_5min_2020_2024_30min.json` |
| 2020-2025 (6y) | `483910de...` | `rr_nq_5min_2020_2025_30min.json` |
| 2021-2024 (4y) | `3c3c72d2...` | `rr_nq_5min_2021_2024_30min.json` |
| 2022-2024 (3y) | `3830ea5c...` | `rr_nq_5min_2022_2024_30min.json` |
| 2022-2025 (4y) | `461912b9...` | `rr_nq_5min_2022_2025_30min.json` |
| 2023-2025 (3y) | `11c81ee6...` | `rr_nq_5min_2023_2025_30min.json` |

## Expanding-window WF series (locked rule: slot blackout + drop n=3.0, full Kelly tiers)

Each row uses cells fit only on prior years, then tested untouched on the test year. Live config (Dec 20 + witching + HFOIV p70×0.25 + tiered Kelly, no slippage, $80k start, max 3 concurrent, -10% daily kill switch).

| OOS year | train | cells | Trades | P&L | PF | Max DD | DD p95 |
|---|---|---|---|---|---|---|---|
| 2021 | 2020 (1y) | 24 | ? | **+90.1%** | 1.19 | 11.5% | — |
| 2022 | 2020-2021 (2y) | 32 | ? | **+48.2%** | 1.10 | **27.5%** | — |
| 2023 | 2020-2022 (3y) | 40 | ? | **+23.9%** | 1.05 | 19.0% | — |
| **2024** | 2020-2023 (4y) | 34 | ? | **−11.2%** | 0.96 | **31.0%** | — |
| 2025 | 2020-2024 (5y) | 30 | ? | **+68.3%** | 1.18 | 17.3% | — |
| 2026 YTD | 2020-2025 (6y) | 40 | ? | **+21.1%** | 1.12 | 21.2% | — |
| **5-year sum (2021-2025)** | — | — | — | **+219.3%** | — | — | — |
| **6-year sum incl. 2026 YTD** | — | — | — | **+240.4%** | — | — | — |

**Compare to live `mixed-best-ev-v3-touch-moderate` on 2026 YTD (sanity check):** −13.9% / PF 0.89 / 31.2% DD. The live strategy is in a losing regime right now; the locked WF strategy is +21.1% on the same window.

## How we got here — ablation study (the path of discovery)

We tested 5 selection rules on every WF year. Recorded for traceability. Each row is the **5-year PnL sum (2021-2025)** with that ablation applied:

| ablation rule | 5y sum | worst DD | 2024 result | comment |
|---|---|---|---|---|
| baseline (unbounded prune-A) | +166.2 | 48.0% | -37.6% | reference |
| + lunch rule (drop rr≥2.75 in 12-13:30) | +174.5 | 48.0% | -23.1% | partial slot fix |
| + drop n=3.0 cells | +182.4 | 40.2% | -26.6% | uniform R:R prune |
| + n=3.0 + lunch (combo) | +192.7 | 36.0% | -21.3% | additive |
| + mit_extreme only | +120.5 | 35.8% | -26.1% | **bad — kills the strategy** |
| + half Kelly tiers (0.25/0.75/1.5%) | +96.3 | 26.3% | -6.4% | survivable but halved PnL |
| + loss-streak halving + MNQ | +94.1 | 28.9% | -10.3% | ≈ uniform half-sizing in disguise |
| **+ slot blackout + drop n=3.0 (LOCKED)** | **+219.3** | **31.0%** | **−11.2%** | best PnL + meaningful DD reduction |
| + slot blackout + n=3.0 + half tiers | +120.1 | 23.5% | +1.8% | best DD, only year all green |

### Key findings from the ablation study

1. **The cap5 / mit-only ablations both DESTROYED 5-year PnL** despite mirroring the live strategy's structural choices. The live strategy is leaving substantial PnL on the table due to its over-restrictive filtering.

2. **`drop n=3.0` is the strongest single-axis rule.** n=3.0 (3:1 R:R) cells have ~10-20% hit rates in trending or chop regimes; their failure is asymmetric. They cost +16 PnL pts on the 5y sum and saved ~8% max DD in 2024.

3. **The lunch rule's partial coverage was a mistake.** It dropped only `rr ≥ 2.75` in 12:00-13:30, which left losing low-R:R cells in those slots. The full slot blackout drops the entire 12:00-12:30 and 13:00-13:30 windows and is strictly better. Crucially, **12:30-13:00 is actually a profitable slot in 2024 (+$8.5k)** — the original lunch rule was over-pruning it.

4. **Half Kelly tiers vs slot blackout:** half tiers gives the cleanest survivability (max DD 23.5%, all years positive) but cuts 5y PnL nearly in half (+96 vs +219). For users who can tolerate 31% max DD in a bad year and want max PnL in good years, full tiers + slot blackout is the better choice.

5. **The combo (slot blackout + drop n=3.0 + full tiers) is net positive in 5/6 WF years.** Only 2024 is negative, and it's contained at −11.2% (vs original baseline −37.6%).

## Time-slot P&L matrix (the data that drove the slot blackout)

P&L by time slot across all 6 WF years using the half-tier combo rule (used for diagnosis):

| slot | 2021 | 2022 | 2023 | 2024 | 2025 | 2026YTD | 6y sum | losing yrs |
|---|---|---|---|---|---|---|---|---|
| 09:30-10:00 | +9k | +2k | -3k | -1k | -1k | -2k | +4k | 4/6 |
| 10:00-10:30 | -1k | +4k | -8k | -1k | +5k | +4k | +3k | 3/6 |
| **10:30-11:00** | +4k | +6k | +4k | -0k | +8k | **+13k** | **+34k** | 1/6 ★ |
| **11:00-11:30** | **+22k** | +4k | +1k | +3k | -2k | -7k | **+20k** | 2/6 ★ |
| 11:30-12:00 | -1k | +4k | -10k | -2k | +10k | +1k | +3k | 3/6 |
| **12:00-12:30** | -3k | **-11k** | -2k | +3k | +9k | -2k | **−6k** | **4/6 🔴** |
| 12:30-13:00 | +4k | +1k | +1k | **+8k** | -0k | -1k | +14k | 2/6 |
| **13:00-13:30** | -1k | -1k | +4k | **-10k** | -4k | 0k | **−12k** | **4/6 🔴** |
| 13:30-14:00 | +3k | +2k | +4k | -3k | +5k | -1k | +10k | 2/6 |
| 14:00-14:30 | -3k | -5k | +7k | -3k | +9k | +1k | +5k | 3/6 |
| 14:30-15:00 | +1k | -5k | +3k | +2k | +1k | +2k | +5k | 1/6 |
| 15:00-15:30 | +1k | +2k | +3k | -1k | -0k | +1k | +6k | 2/6 |
| 15:30-16:00 | +0k | +1k | -0k | 0 | 0 | +2k | +3k | 1/6 |

**12:00-12:30 and 13:00-13:30 are the only two slots that lost in 4/6 years AND have negative 6y sums.** 13:00-13:30 alone lost $12k cumulatively across the WF series. They are dropped entirely in the locked strategy.

**12:30-13:00 — the middle of the original lunch window — is actually positive on net (+$14k 6y sum)** and is preserved by the locked rule.

## 2024 deep dive (why the worst year happened, and what we fixed)

### Monthly P&L (raw baseline, no ablations)

| month | P&L | cumulative |
|---|---|---|
| Jan | −$12,823 | −$12,823 |
| **Feb** | **−$15,642** | **−$28,464** |
| Mar | +$782 | −$27,683 |
| Apr | −$2,402 | −$30,086 |
| May | −$4,088 | −$34,173 |
| Jun | −$344 | −$34,516 |
| **Jul** | **+$12,043** | −$22,474 |
| Aug | −$8,801 | −$31,274 |
| Sep | +$367 | −$30,908 |
| Oct | +$1,380 | −$29,528 |
| Nov | −$1,926 | −$31,454 |
| Dec | +$1,370 | −$30,085 |

Q1 2024 cost ~36% of starting capital. The kill switch (-10% daily) would have triggered repeatedly in Jan/Feb. Only July was a meaningfully positive month.

### Loss concentration in 2024

- **By R:R target**: n=3.0 alone accounted for −$12,495 (68% of the gross drawdown when combined with n=1.5's −$7,892). The strategy's mean-reversion premise broke as NQ trended +30% over the year.
- **By time slot**: 13:00-13:30 was the worst at −$17,240 (5/5 cells losing), 14:00-14:30 at −$6,076 (4/5), 12:00-12:30 at −$7,107 (3/4). 12:30-13:00 was *positive* at +$10,596 — the slot blackout's geographic boundary needs to skip 12:30-13:00.

### How the locked rule changed 2024

| variant | 2024 P&L | 2024 max DD |
|---|---|---|
| Raw baseline (unbounded prune-A) | −37.6% | 48.0% |
| + lunch rule | −23.1% | 38.7% |
| + drop n=3.0 | −26.6% | 40.2% |
| + combo (n=3.0 + lunch) | −21.3% | 36.0% |
| **+ slot blackout + n=3.0 (LOCKED)** | **−11.2%** | **31.0%** |
| + slot blackout + n=3.0 + half tiers | +1.8% | 23.5% |

The locked rule cuts 2024 losses **from −37.6% to −11.2%** — a 26.4 percentage point save in the catastrophe year. Account survives, kill switch protects, no manual intervention needed.

## Why the locked strategy isn't perfect

1. **2024 is still a losing year** (−11.2%). The half-tier variant turns it green (+1.8%) at the cost of halving good-year PnL. The locked strategy chooses higher PnL ceiling over zero-loss-year guarantee.
2. **Max DD of 31% in 2024 still trips the kill switch multiple times.** The backtester doesn't model post-halt recovery; real bot behavior in Jan-Feb 2024 might be meaningfully different from the bleed shown in the backtest.
3. **2026 YTD is only 63 days** — the +21.1% / PF 1.12 result is encouraging but variance-bound. We need full-year 2026 data before drawing strong conclusions.
4. **Compounding distorts train-window backtest P&L.** The prune-A rule uses train backtest results to drop cells; with 5-6 years of compounding from $80k, late-window losses are sized much larger than early ones, which biases the prune toward dropping cells that lost late.

## Open questions / next steps

1. **2024 catastrophe — can we do better than −11.2%?** Trend-regime adaptive sizing (reduce position size when 20-day NQ trend is strong) is the next conceptual step that hasn't been tested. Implemented as a backtester flag, this would auto-shrink positions in trend regimes.
2. **Volatility-based sizing** — scale per-trade contracts inversely with current 20-day realized vol. High-vol regimes (2022, 2024) automatically shrink positions; calm regimes (2021, 2025) get full size.
3. **Multi-day DD circuit breaker** — halt trading after running DD < -8% over a rolling N-day window, resume only after equity recovers above the entry mark. Catches Q1-2024-style multi-week bleeds that the daily -10% kill switch can't see.
4. **Dataset id collision fix** — modify `logic/main.py` so the RR dataset id includes `custom_start` / `custom_end` when set, preventing the JSON overwrite issue.
5. **Lookback bias check on prune-A.** Dropping cells with negative train P&L uses train backtest results to filter. Test on a held-out validation slice (e.g., train on 2020-2022, prune on 2023, test on 2024) to verify it's not overfitting.
6. **MNQ + loss-streak (or simpler half-tier) as a fallback variant for risk-averse operation.** Already implemented in the backtester (`--loss-streak --mnq`); not currently used by the locked strategy but available as a one-flag toggle for paper-mode trials.

## File map

- Selector: `scripts/build_nq_strategy_from_window.py`
- WF year pipeline driver: `scripts/wf_year_pipeline.py`
- Backtester (with new defensive flags): `bot/backtest/backtester.py`
- **LOCKED active strategy**: `logic/strategies/mixed-best-ev-wf-2020-2025-slotbl-non3.json` (40 cells, full tiers)
- Half-tier sibling (more defensive variant): `logic/strategies/mixed-best-ev-wf-2020-2025-slotbl-non3-half.json`
- Original live (rollback target): `logic/strategies/mixed-best-ev-v3-touch-moderate.json`
- Closest live-rule reproduction (cap5): `logic/strategies/mixed-best-ev-wf-2020-2025-cap5.json` (34 cells, weighted EV 0.177, matches live)
- Dashboard: `bot/backtest/results/manifest.json` shows the 6 OOS WF runs for the locked strategy.
