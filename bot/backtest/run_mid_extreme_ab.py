#!/usr/bin/env python
"""A/B comparison: baseline vs --skip-mid-if-mit using exact WF backtest configs."""

import subprocess
import json
import sys
import os

# Exact walk-forward configs from 20260408_181720 series
WF_YEARS = [
    # (label, start, end, strategy_id, balance, margin)
    ("2021", "20210105", "20211215", "wf-2020-test-2021-slotbl-non3", 80000.0, 33000.0),
    ("2022", "20220103", "20221219", "wf-2020-2021-test-2022-slotbl-non3", 80000.0, 33000.0),
    ("2023", "20230103", "20231219", "wf-2020-2022-test-2023-slotbl-non3", 80000.0, 33000.0),
    ("2024", "20240102", "20241218", "wf-2020-2023-test-2024-slotbl-non3", 80000.0, 33000.0),
    ("2025", "20250102", "20251217", "mixed-best-ev-wf-2020-2024-slotbl-non3", 80000.0, 33000.0),
    ("2026 YTD", "20260102", "20260407", "mixed-best-ev-wf-2020-2025-slotbl-non3", 80000.0, 33000.0),
]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

rows = []
for label, start, end, strategy, balance, margin in WF_YEARS:
    for variant, extra_flags in [("baseline", []), ("skip_mid", ["--skip-mid-if-mit"])]:
        out_file = os.path.join(RESULTS_DIR, f"ab_mid_{label.replace(' ', '_')}_{variant}.json")
        cmd = [
            sys.executable, "-m", "bot.backtest.backtester",
            "--strategy", strategy,
            "--balance", str(balance),
            "--margin", str(margin),
            "--risk-tiers",
            "--hfoiv", "--hfoiv-rolling", "6", "--hfoiv-lookback", "90",
            "--start", start, "--end", end,
            "--json-output", out_file,
        ] + extra_flags
        print(f"\n{'='*60}")
        print(f"  {label} — {variant} (strategy={strategy})")
        print(f"{'='*60}")
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"  FAILED (rc={result.returncode})")
            continue
        try:
            with open(out_file) as f:
                data = json.load(f)
            s = data["summary"]
            rows.append({
                "year": label,
                "variant": variant,
                "trades": s["total_trades"],
                "win_rate": s["win_rate"],
                "net_pnl": s["net_pnl"],
                "return_pct": s.get("pnl_pct", 0),
                "max_dd": s.get("max_dd_pct", 0),
                "profit_factor": s.get("profit_factor", 0),
                "final_balance": s["final_balance"],
            })
        except Exception as e:
            print(f"  Error reading results: {e}")

# Print comparison table
print("\n" + "=" * 110)
print("  A/B COMPARISON: baseline vs skip-mid-if-mit (WF configs)")
print("=" * 110)
print(f"{'Year':<12} {'Variant':<12} {'Trades':>7} {'WinRate':>8} {'Net PnL':>14} {'Return%':>9} {'MaxDD%':>8} {'PF':>6}")
print("-" * 110)
prev_year = None
for r in rows:
    if prev_year and r['year'] != prev_year and r['variant'] == 'baseline':
        print()
    prev_year = r['year']
    print(f"{r['year']:<12} {r['variant']:<12} {r['trades']:>7} {r['win_rate']:>7.1f}% {r['net_pnl']:>13,.0f} {r['return_pct']:>8.1f}% {r['max_dd']:>7.1f}% {r['profit_factor']:>6.2f}")
print("-" * 110)

# Totals
for variant in ["baseline", "skip_mid"]:
    vrows = [r for r in rows if r["variant"] == variant]
    if vrows:
        total_trades = sum(r["trades"] for r in vrows)
        total_pnl = sum(r["net_pnl"] for r in vrows)
        avg_wr = sum(r["win_rate"] * r["trades"] for r in vrows) / total_trades if total_trades else 0
        worst_dd = max(r["max_dd"] for r in vrows)
        print(f"{'TOTAL':<12} {variant:<12} {total_trades:>7} {avg_wr:>7.1f}% {total_pnl:>13,.0f} {'':>9} {worst_dd:>7.1f}%")

# Delta row
base_rows = [r for r in rows if r["variant"] == "baseline"]
skip_rows = [r for r in rows if r["variant"] == "skip_mid"]
if base_rows and skip_rows:
    print()
    print(f"{'DELTA':<12} {'':>12} {sum(r['trades'] for r in skip_rows) - sum(r['trades'] for r in base_rows):>+7} {'':>8} {sum(r['net_pnl'] for r in skip_rows) - sum(r['net_pnl'] for r in base_rows):>+13,.0f}")
