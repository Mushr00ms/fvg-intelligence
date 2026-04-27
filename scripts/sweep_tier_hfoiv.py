#!/usr/bin/env python3
"""
Sweep risk tier maps across all years (2021-2026 YTD), with and without HFOIV.
Tests 5 tier configurations × 6 years × 2 HFOIV settings = 60 runs.
Uses walk-forward strategy files (each year trained only on prior data).
"""
import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STRAT_DIR = ROOT / "bot" / "strategies"

# Walk-forward strategy mapping: each year uses strategy trained on prior data only
YEAR_CONFIG = {
    2021: {"file": "wf-2020-test-2021-slotbl-non3.json",           "start": "20210101", "end": "20211231"},
    2022: {"file": "wf-2020-2021-test-2022-slotbl-non3.json",      "start": "20220101", "end": "20221231"},
    2023: {"file": "wf-2020-2022-test-2023-slotbl-non3.json",      "start": "20230101", "end": "20231231"},
    2024: {"file": "wf-2020-2023-test-2024-slotbl-non3.json",      "start": "20240101", "end": "20241231"},
    2025: {"file": "mixed-best-ev-wf-2020-2024-slotbl-non3.json",  "start": "20250101", "end": "20251231"},
    2026: {"file": "mixed-best-ev-wf-2020-2025-slotbl-non3.json",  "start": "20260101", "end": "20260407"},
}

# Tier maps to test: (label, small_pct, medium_pct, large_pct)
TIER_MAPS = [
    ("0.50/1.50/3.00 (baseline)", 0.005, 0.015, 0.03),
    ("0.25/1.00/3.00",            0.0025, 0.01, 0.03),
    ("0.25/0.75/3.00",            0.0025, 0.0075, 0.03),
    ("0.50/0.75/3.00",            0.005, 0.0075, 0.03),
    ("0.25/0.75/2.00",            0.0025, 0.0075, 0.02),
]


def make_temp_strategy(original_path: Path, small: float, medium: float, large: float) -> str:
    """Create a temp strategy file with modified risk tiers. Returns path."""
    with open(original_path) as f:
        strat = json.load(f)

    strat["meta"]["risk_rules"]["small_risk_pct"] = small
    strat["meta"]["risk_rules"]["medium_risk_pct"] = medium
    strat["meta"]["risk_rules"]["large_risk_pct"] = large

    fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix="tier_sweep_")
    with os.fdopen(fd, "w") as f:
        json.dump(strat, f)
    return tmp_path


def run_backtest(strategy_path: str, start: str, end: str, hfoiv: bool) -> dict:
    """Run backtester and parse JSON output."""
    fd, out_path = tempfile.mkstemp(suffix=".json", prefix="bt_result_")
    os.close(fd)

    cmd = [
        sys.executable, str(ROOT / "bot" / "backtest" / "backtester.py"),
        "--strategy-file", strategy_path,
        "--start", start, "--end", end,
        "--risk-tiers",
        "--balance", "80000",
        "--risk-pct", "0.01",
        "--margin", "1",
        "--json-output", out_path,
    ]
    if hfoiv:
        cmd.extend(["--hfoiv", "--hfoiv-rolling", "6", "--hfoiv-lookback", "90"])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            # Try to extract key info from stderr
            err_lines = (proc.stderr or "").strip().split("\n")[-5:]
            return {"error": "\n".join(err_lines)}

        with open(out_path) as f:
            return json.load(f)
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


def extract_metrics(result: dict) -> dict:
    """Extract P&L, DD, PF from backtest result."""
    if "error" in result:
        return {"pnl": "ERR", "pnl_pct": "ERR", "max_dd": "ERR", "pf": "ERR", "trades": "ERR"}

    summary = result.get("summary", result)
    pnl = summary.get("net_pnl", 0)
    pnl_pct = summary.get("pnl_pct", 0)
    max_dd = summary.get("max_dd_pct", 0)
    pf = summary.get("profit_factor", 0)
    trades = summary.get("total_trades", 0)

    return {
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "max_dd": max_dd,
        "pf": pf,
        "trades": trades,
    }


def main():
    results = {}  # {(tier_label, year, hfoiv_on): metrics}
    total_runs = len(TIER_MAPS) * len(YEAR_CONFIG) * 2
    run_num = 0

    for tier_label, small, medium, large in TIER_MAPS:
        for year, cfg in sorted(YEAR_CONFIG.items()):
            strat_path = STRAT_DIR / cfg["file"]
            if not strat_path.exists():
                print(f"  SKIP: {strat_path.name} not found")
                continue

            tmp_strat = make_temp_strategy(strat_path, small, medium, large)
            try:
                for hfoiv in [False, True]:
                    run_num += 1
                    hfoiv_label = "HFOIV" if hfoiv else "noHFOIV"
                    print(f"[{run_num}/{total_runs}] {tier_label} | {year} | {hfoiv_label} ...", flush=True)

                    result = run_backtest(tmp_strat, cfg["start"], cfg["end"], hfoiv)
                    metrics = extract_metrics(result)
                    results[(tier_label, year, hfoiv)] = metrics

                    if metrics["pnl"] == "ERR":
                        print(f"  ERROR: {result.get('error', 'unknown')}")
                    else:
                        print(f"  P&L: ${metrics['pnl']:,.1f} ({metrics['pnl_pct']:+.1f}%) | DD: {metrics['max_dd']:.1f}% | PF: {metrics['pf']:.2f} | Trades: {metrics['trades']}")
            finally:
                os.unlink(tmp_strat)

    # Print summary tables
    print("\n" + "=" * 120)
    print("FULL RESULTS SUMMARY")
    print("=" * 120)

    for hfoiv in [False, True]:
        hfoiv_label = "WITH HFOIV" if hfoiv else "WITHOUT HFOIV"
        print(f"\n{'─' * 120}")
        print(f"  {hfoiv_label}")
        print(f"{'─' * 120}")
        header = f"{'Tier Map':<28} | {'Year':>4} | {'P&L':>12} | {'P&L %':>8} | {'Max DD':>8} | {'PF':>6} | {'Trades':>6}"
        print(header)
        print("─" * 120)

        for tier_label, _, _, _ in TIER_MAPS:
            for year in sorted(YEAR_CONFIG.keys()):
                key = (tier_label, year, hfoiv)
                m = results.get(key)
                if not m:
                    continue
                if m["pnl"] == "ERR":
                    print(f"{tier_label:<28} | {year:>4} | {'ERROR':>12} | {'':>8} | {'':>8} | {'':>6} | {'':>6}")
                else:
                    print(f"{tier_label:<28} | {year:>4} | ${m['pnl']:>10,.1f} | {m['pnl_pct']:>+7.1f}% | {m['max_dd']:>7.1f}% | {m['pf']:>5.2f} | {m['trades']:>6}")
            print()

    # Side-by-side HFOIV comparison
    print(f"\n{'=' * 140}")
    print("HFOIV IMPACT (delta = WITH minus WITHOUT)")
    print(f"{'=' * 140}")
    header2 = f"{'Tier Map':<28} | {'Year':>4} | {'noHFOIV P&L%':>12} | {'noHFOIV DD':>10} | {'HFOIV P&L%':>10} | {'HFOIV DD':>8} | {'ΔP&L%':>7} | {'ΔDD':>7}"
    print(header2)
    print("─" * 140)

    for tier_label, _, _, _ in TIER_MAPS:
        for year in sorted(YEAR_CONFIG.keys()):
            m_off = results.get((tier_label, year, False))
            m_on = results.get((tier_label, year, True))
            if not m_off or not m_on or m_off["pnl"] == "ERR" or m_on["pnl"] == "ERR":
                continue
            d_pnl = m_on["pnl_pct"] - m_off["pnl_pct"]
            d_dd = m_on["max_dd"] - m_off["max_dd"]
            print(f"{tier_label:<28} | {year:>4} | {m_off['pnl_pct']:>+11.1f}% | {m_off['max_dd']:>9.1f}% | {m_on['pnl_pct']:>+9.1f}% | {m_on['max_dd']:>7.1f}% | {d_pnl:>+6.1f}% | {d_dd:>+6.1f}%")
        print()

    # Save raw results
    out_file = ROOT / "scripts" / "tier_hfoiv_sweep_results.json"
    serializable = {}
    for (t, y, h), m in results.items():
        key_str = f"{t}|{y}|{'hfoiv' if h else 'nohfoiv'}"
        serializable[key_str] = m
    with open(out_file, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nRaw results saved to {out_file}")


if __name__ == "__main__":
    main()
