# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FVG Intelligence is a financial analysis system for detecting and analyzing Fair Value Gaps (FVGs) in futures markets (NQ, ES). It provides multi-timeframe analysis pipelines, risk-reward trade simulation, and interactive web dashboards built on FastAPI + Plotly.

## Commands

```bash
# Start the API server (serves dashboards + API on port 8765)
python api/heatmap_api.py

# Run analysis pipeline (configured via environment variables)
FVG_TICKER=NQ FVG_TIMEFRAME=5min python logic/main.py

# Regenerate all pre-computed datasets
python regen_all.py          # Sequential
python regen_parallel.py     # Parallel (4 workers)
python regen_3min.py         # 3-minute timeframe only

# Install dependencies (hardened — uses local wheel cache + hash verification)
bash scripts/check_deps.sh
# Or manual: pip install --require-hashes --no-deps --no-index --find-links vendor/wheels -r requirements.lock
```

### Key Environment Variables for `logic/main.py`

| Variable | Default | Values |
|----------|---------|--------|
| `FVG_TICKER` | NQ | NQ, ES |
| `FVG_PERIOD` | "5 years" | e.g. "1 year", "15 years" |
| `FVG_TIMEFRAME` | (all three) | 3min, 5min, 15min |
| `FVG_SESSION_MINUTES` | (from config) | e.g. 30, 60 |
| `FVG_METHOD` | bins | bins, cumulative |
| `FVG_MIN_EXP` | (from config) | Minimum expansion size |
| `FVG_SIZE_START/END/STEP` | (from config) | Size range overrides |

## Architecture

```
Frontend (heatmap_dashboard/)    →  3 dashboards: heatmap workspace, R:R explorer, live monitor
    ↕ HTTP/REST + SSE
API (api/heatmap_api.py)         →  FastAPI server on :8765, spawns analysis as subprocesses
    ↕ subprocess + env vars
Engine (logic/main.py)           →  Orchestrates 3 parallel pipelines (3min/5min/15min)
    ↕
Core (logic/utils/)              →  Detection, mitigation, expansion, RR simulation, stats
    ↕
Data (logic/heatmap_data/, rr_data/, fvg_cache/, databento_cache/)

Bot (bot/)                       →  Live/paper trading via IB TWS (separate subsystem)
    ├── core/engine.py           →  FVG detection → mitigation scan → order placement
    ├── risk/                    →  Risk gates (5 checks) + time gates (EOD schedule)
    ├── execution/               →  Bracket orders via OCA groups, partial fill handling
    ├── state/                   →  Atomic JSON persistence, crash recovery
    ├── backtest/                →  1-second bar replay, minute-by-minute fill simulation
    └── strategy/                →  Cell lookup (time × risk → setup), hot-reload every 60s
```

### Bot System (`bot/`)

The trading bot connects to Interactive Brokers TWS via `ib_async`. It runs in **paper mode** (port 7497) by default; live mode (port 7496) requires explicit `--live --no-dry-run` flags.

**Core loop** (`bot/core/engine.py`): Detect FVGs from 5min bars → scan for mitigation → check risk/time gates → place bracket orders (entry + TP + SL via OCA groups).

**Order lifecycle**: Bracket orders use pre-allocated IB order IDs (saved to state BEFORE placement for crash safety). Partial fills trigger a 5-minute timer; unfilled remainder is cancelled and TP/SL adjusted to match filled quantity.

**EOD schedule** (hard-coded ET in `bot/risk/time_gates.py`):
- Entry window: 09:30–15:30
- Cancel unfilled: 15:50
- Flatten all positions: 15:55

**Risk gates** (`bot/risk/risk_gates.py`) — checked in order, first failure blocks trade:
1. Kill switch active? (-3% daily loss → flatten all + halt)
2. Daily trade count ≤ 15
3. Concurrent positions ≤ 3
4. Per-trade risk ≤ 1% of balance
5. Max trade loss ≤ 1.5% (with slippage buffer)

**Crash recovery**: Atomic state persistence to `bot/bot_state.json` via `.tmp` + `os.replace()`, debounced to max 1 save/sec. On startup, loads state if date matches and reconciles with IB positions. FVGs backfilled from 5min bars.

**NTP clock** (`bot/clock.py`): Syncs against multiple NTP servers (Google, Cloudflare, NIST) to mitigate WSL2 clock drift. 500ms warn threshold, 5s critical threshold, 5-minute re-sync interval.

**WSL2 bridge** (`bot/bridge/ib_bridge.py`): Separate Windows process on port 9100, newline-delimited JSON over TCP. Auto-launched if `auto_launch_bridge=True`.

### Strategy System

Strategies live in `logic/strategies/` as JSON files. Active strategy set in `logic/strategies/manifest.json`. Bot checks file mtime every 60 seconds and swaps lookup atomically (hot-reload).

**Cell structure** — each cell is a pre-approved time-risk-setup triplet:
```json
{"time_period": "10:30-11:00", "risk_range": "10-15", "setup": "mit_extreme", "best_n": 2.75, "ev": 0.1895, "samples": 309}
```

**Lookup**: `StrategyLoader` builds O(1) dict: `{(time_period, risk_range): cell_config}`. Filter criteria: 200+ samples, positive EV at all 9 R:R ratios (1.0–3.0R).

**Risk bucketing**: `risk_to_range()` maps risk points to buckets (5-10, 10-15, ... 40-80). Risk exactly at a boundary (e.g., 10pt) maps to the higher bucket (10-15).

### Pipeline per timeframe

1. **Detect** FVGs by gap size (`fvg_analysis.py`)
2. **Mitigate** — find first price touch after formation
3. **Expand** — track expansion after mitigation
4. **RR Simulate** — 4 entry setups × 9 R:R ratios (`rr_analysis.py`)
5. **Analyze** — size-time distribution grid (`fvg_analysis.py`)
6. **Visualize + Export** — heatmaps, charts, JSON/CSV

### Key API Endpoints

**Analysis**:
- `POST /api/compute` — spawn analysis job (env vars in body)
- `GET /api/jobs/{id}/stream` — SSE stream of job stdout
- `DELETE /api/jobs/{id}` — cancel running job
- `GET /api/dataset/{id}` — load pre-computed heatmap data
- `GET /api/rr/{id}` — load risk-reward data
- `GET /api/rr/trade-sample` — specific trade with candle data
- `GET /api/manifest` / `GET /api/rr/manifest` — dataset indexes
- `GET /api/overview-stats` — aggregate stats across all datasets
- `GET /api/configs` — NQ + ES config JSONs

**Strategy CRUD**:
- `GET /api/strategies` — list all strategies
- `GET /api/strategies/{id}` — load strategy
- `GET /api/strategies/active` — get active strategy
- `POST /api/strategies` — save strategy
- `DELETE /api/strategies/{id}` — delete strategy
- `PUT /api/strategies/active/{id}` — set active strategy

**Backtest**:
- `POST /api/backtest` — run backtest
- `GET /api/backtest/{id}/stream` — backtest progress SSE
- `GET /api/backtest/{id}/results` — get results
- `GET /api/backtest/history` — list all backtests

**Bot** (live monitoring):
- `GET /api/bot/state` — current bot state
- `GET /api/bot/events` — SSE stream of bot events
- `GET /api/bot/events/history` — event history
- `GET /api/bot/stats` — bot statistics

**Dashboards**: `GET /` (index), `GET /heatmap`, `GET /rr`, `GET /live`

### Configuration

Market-specific configs live in `logic/configs/{NQ,ES}_config.json` with per-timeframe FVG detection thresholds, session times, size filtering methods, and contract specs. `logic/config.py` loads these and applies environment variable overrides.

### Data & Caching

- **FVG cache**: MD5-hashed parquet files in `logic/fvg_cache/` (invalidated on param change)
- **Market data**: Parquet + in-memory cache with 4GB limit in `logic/databento_data/`
- **Heatmap store**: Column-oriented JSON (schema v1.0) in `logic/heatmap_data/` with `manifest.json` index
- **RR data**: JSON in `logic/rr_data/` with its own manifest
- **Backtest data**: 1-second bar parquets (`nq_1secs_YYYYMMDD.parquet`) resampled on-the-fly to 5min for FVG detection

### Databento Tiered Cache

`DatabentoCache` (`logic/utils/databento_cache_manager.py`) implements 4 tiers in `logic/databento_cache/`:

| Tier | Period | Compression | Priority |
|------|--------|-------------|----------|
| TIER1 | 2yr | Snappy | 1 (fastest) |
| TIER2 | 5yr | Gzip | 2 |
| TIER3 | 10yr | Brotli | 3 |
| TIER4 | 15yr | Brotli | 4 |

Cache metadata persists in `logic/databento_cache/cache_metadata.json`. Contract mapping (`DatabentoContractMapper`) resolves Databento symbols to IB symbols.

### FVG Detection Result Tuple

```python
(fvg_type, y0, y1, time_candle1, time_candle2, time_candle3,
 idx, middle_open, middle_low, middle_high, first_open)
```

This tuple structure is used throughout the codebase — any changes to it require updating all consumers in `fvg_analysis.py`, `rr_analysis.py`, and `visualization_utils.py`.

### Key Gotchas

1. **Timezone handling**: IB bars arrive in CME/Central time (naive). Bot converts to ET via `_bar_date_to_et()` in `bot/core/engine.py`. All session times (15:50, 15:55 flatten) are hard-coded ET.
2. **Slippage model**: Backtester-only. Entry: 1 tick deeper into zone. Target: 1 tick early. Stop: exact. Live bot places limit orders at exact prices — no slippage modeling.
3. **Session period minutes**: 30min sessions for 3min/5min timeframes (~18-29 cells), 60min sessions for 15min timeframe (fewer, larger cells). Affects time_period binning (e.g., "10:30-11:00" vs "10:30-11:30").
4. **Partial fill complexity**: 5-minute timer after partial entry fill. Unfilled remainder cancelled. TP/SL adjusted to match filled qty. `OrderGroup.filled_qty` tracked separately from `target_qty`.
5. **FVG expiration**: Same-day only. Un-mitigated FVGs discarded at 16:00 ET. On crash restart, FVGs backfilled from 5min historical bars.
6. **Contract resolution**: Tries YYYYMMDD then YYYYMM format for newly-listed contracts. Suppresses IB error 200 during probing. Roll logic uses `generate_nq_expirations()` based on `roll_days` config param.
7. **Macro events**: `logic/configs/macro_events.json` contains NFP (Non-Farm Payroll) dates 2020-2025 for high-volatility day filtering.

## Tech Stack

Python 3 (FastAPI, Pandas, NumPy, Matplotlib, Plotly, PyArrow), Databento for market data, frontend is vanilla HTML/CSS/JS with Plotly.js.

### Tests

11 pytest files in `tests/` covering bot subsystems: FVG detection, mitigation, risk gates, time gates, state management, strategy loader, position tracker, trade calculator, clock, strategy store. Run with:

```bash
python -m pytest tests/ -v
```

---

## Working Protocols

The sections below define operational rules for how Claude Code should work in this codebase. They are based on [Anthropic's harness design patterns](https://www.anthropic.com/engineering/harness-design-long-running-apps) for long-running agent tasks, adapted for interactive use.

### Task Decomposition

Before starting any non-trivial task:

1. **State acceptance criteria.** Define what "done" looks like in concrete, verifiable terms before writing any code. Get user confirmation.
2. **Atomic steps.** Break multi-file changes into steps where each step leaves the codebase in a working state (tests pass, no syntax errors). Never leave the repo broken between steps.
3. **Plan-first for broad changes.** If a task touches 3+ files, write a brief plan listing files and changes before executing. Get approval.
4. **Size to one context window.** A single work sprint should be completable without context compaction. If a task looks larger, split it upfront and define handoff points between sprints.
5. **Tests before features.** When adding new functionality, write or update tests first to define expected behavior, then implement.

### Quality Gates

Before marking any task as done, run these mechanical checks. Do not skip them even for "small" changes.

- [ ] **Tests pass**: `python -m pytest tests/ -v` — zero failures
- [ ] **Syntax clean**: `python -m py_compile <changed_file>` on every modified Python file
- [ ] **FVG tuple consistency**: If anything in `logic/utils/` was changed, verify the FVG detection result tuple is consumed identically across `fvg_analysis.py`, `rr_analysis.py`, and `visualization_utils.py`
- [ ] **Dependency pipeline intact**: If `requirements.in` was changed, `requirements.lock` must be regenerated and `bash scripts/check_deps.sh` must pass
- [ ] **No deleted tests**: Never delete a test without explicit user approval and a documented reason
- [ ] **No skipped tests**: Never mark a test as `@pytest.mark.skip` to make the suite pass
- [ ] **Bot safety invariants**: If `bot/` was changed — verify risk gate defaults unchanged (`max_concurrent=3`, `max_daily_trades=15`, `kill_switch_pct=-0.03`), time gate constants unchanged (cancel 15:50, flatten 15:55), and bracket order/OCA logic intact
- [ ] **No time gate modifications**: Never change EOD schedule constants (15:50 cancel, 15:55 flatten, 16:00 session end) without explicit approval
- [ ] **Price calculation parity**: If order/execution/RR code changed, compare numerical outputs before and after

If a test fails after your changes: fix the code or fix the test (with explanation). Do not proceed to the next step until the suite is green.

### Context Management

#### Checkpoint triggers

Create a handoff file when any of these occur:
- Task is partially complete and the session is ending
- Switching between unrelated subsystems (e.g., `logic/` work to `bot/` work)
- A complex task spans multiple conversation turns and you risk losing state

#### Handoff file convention

**Location**: `.claude/handoffs/YYYY-MM-DD-HHMM-kebab-description.md`

**Format**:
```markdown
# Handoff: [short description]
## Status: [IN_PROGRESS | BLOCKED | REVIEW_NEEDED]
## What was done
- [completed changes with file paths]
## What remains
- [remaining work items]
## Current state
- Branch: [branch name]
- Last commit: [hash + message]
- Tests passing: [yes/no + details]
## Decisions made
- [architectural decisions with rationale]
## Open questions
- [anything needing user input]
```

#### On session start

Check `.claude/handoffs/` for recent files. If one exists with status `IN_PROGRESS`, read it and continue from where it left off. Do not re-explore code that was already explored in the previous session.

#### Cleanup

Delete handoff files once their task is complete. Maximum 5 active handoff files — clean up oldest completed ones if more exist.

### Financial Safety Protocol

This codebase manages real trading capital. These rules are non-negotiable.

1. **Never start the bot in live mode.** The `--live` and `--no-dry-run` flags are blocked in `.claude/settings.json`. Do not attempt to circumvent this. Paper mode (port 7497) only.
2. **Never modify risk gate thresholds** (`max_concurrent`, `max_daily_trades`, `kill_switch_pct`, per-trade risk limits) without explicit user approval. These exist to prevent catastrophic loss.
3. **Never remove or weaken kill switch logic.** Even if it appears redundant or overly conservative.
4. **Changes to safety-critical paths require risk statement BEFORE implementation.** These paths are: `bot/risk/`, `bot/execution/`, `bot/core/engine.py`, `bot/state/`, and any price/order calculation code. State what could go wrong, what the worst case is, and how you've mitigated it.
5. **All price calculation changes require before/after comparison.** Run the relevant test or produce sample outputs showing the change is numerically correct.

### Anti-Drift Rules

These rules prevent the most common failure modes in long-running agent work.

1. **Do not simplify the user's request.** If asked to implement X, implement X completely. Do not implement a subset and call it done.
2. **Do not silently reduce scope.** If you discover the task is harder than expected, say so and propose a revised plan. Do not quietly deliver less than what was agreed.
3. **Do not skip validation to save time.** Running tests is not optional. Running py_compile is not optional. The quality gates exist because skipping them has caused real bugs.
4. **Financial system awareness.** This codebase manages real trading capital. Changes to `bot/risk/`, `bot/execution/`, `bot/state/`, and any price/order calculation code require extra scrutiny. Explain risk implications of any changes to these subsystems before implementing.
5. **Preserve existing behavior.** When refactoring, verify that outputs do not change. For numerical code (FVG detection, RR simulation), even small floating-point drift matters. Compare outputs before and after.
6. **No placeholder implementations.** Every function must be complete and tested, or explicitly marked as `TODO` with a justification agreed upon with the user.
7. **No unrelated cleanup.** Do not refactor, reformat, or "improve" code that is not part of the current task. Scope discipline prevents regressions.

### Progress Tracking

For multi-step tasks:
1. Maintain a step checklist in the conversation showing completed vs remaining steps
2. After each step, briefly state what was done and what is next
3. If a step takes longer than expected or the approach needs to change, explain before continuing — do not silently pivot

### Error Recovery

1. **Failing tests block forward progress.** Do not move to the next step until the test suite is green.
2. **Revert rather than compound.** If you break something that is hard to fix, `git stash` or `git checkout` back to the last good state and retry the step cleanly. Do not layer fixes on top of broken state.
3. **Diagnose, don't workaround.** If a dependency install fails, diagnose the failure — do not remove the dependency. If a test fails, find the root cause — do not delete the test.
4. **API server debugging.** If the API server crashes during testing, check logs and the traceback before retrying. Do not restart blindly.
5. **Bot state corruption.** If modifying state code, read `bot/bot_state.json` structure first. State uses atomic writes (`.tmp` + `os.replace()`); do not bypass this pattern.
6. **IB connection issues.** Check bridge status (port 9100), IB port config (7497 paper / 7496 live), and NTP sync before assuming code bugs.
