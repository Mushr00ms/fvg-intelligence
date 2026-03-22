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

# Install dependencies
pip install -r requirements.txt
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
Frontend (heatmap_dashboard/)    →  HTML + Plotly.js dashboards
    ↕ HTTP/REST + SSE
API (api/heatmap_api.py)         →  FastAPI server on :8765, spawns analysis as subprocesses
    ↕ subprocess + env vars
Engine (logic/main.py)           →  Orchestrates 3 parallel pipelines (3min/5min/15min)
    ↕
Core (logic/utils/)              →  Detection, mitigation, expansion, RR simulation, stats
    ↕
Data (logic/heatmap_data/, rr_data/, fvg_cache/, databento_data/)
```

### Pipeline per timeframe

1. **Detect** FVGs by gap size (`fvg_analysis.py`)
2. **Mitigate** — find first price touch after formation
3. **Expand** — track expansion after mitigation
4. **RR Simulate** — 4 entry setups × 9 R:R ratios (`rr_analysis.py`)
5. **Analyze** — size-time distribution grid (`fvg_analysis.py`)
6. **Visualize + Export** — heatmaps, charts, JSON/CSV

### Key API Endpoints

- `POST /api/compute` — spawn analysis job (configured via posted env vars)
- `GET /api/jobs/{id}/stream` — SSE stream of job stdout
- `GET /api/dataset/{id}` — load pre-computed heatmap data
- `GET /api/rr/{id}` — load risk-reward data
- `GET /api/manifest` / `GET /api/rr/manifest` — dataset indexes

### Configuration

Market-specific configs live in `logic/configs/{NQ,ES}_config.json` with per-timeframe FVG detection thresholds, session times, size filtering methods, and contract specs. `logic/config.py` loads these and applies environment variable overrides.

### Data & Caching

- **FVG cache**: MD5-hashed parquet files in `logic/fvg_cache/` (invalidated on param change)
- **Market data**: Parquet + in-memory cache with 4GB limit in `logic/databento_data/`
- **Heatmap store**: Column-oriented JSON (schema v1.0) in `logic/heatmap_data/` with `manifest.json` index
- **RR data**: JSON in `logic/rr_data/` with its own manifest

### FVG Detection Result Tuple

```python
(fvg_type, y0, y1, time_candle1, time_candle2, time_candle3,
 idx, middle_open, middle_low, middle_high, first_open)
```

This tuple structure is used throughout the codebase — any changes to it require updating all consumers in `fvg_analysis.py`, `rr_analysis.py`, and `visualization_utils.py`.

## Tech Stack

Python 3 (FastAPI, Pandas, NumPy, Matplotlib, Plotly, PyArrow), Databento for market data, frontend is vanilla HTML/CSS/JS with Plotly.js. No test suite or linting config exists.
