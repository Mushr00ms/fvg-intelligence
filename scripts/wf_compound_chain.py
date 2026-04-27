#!/usr/bin/env python3
"""
wf_compound_chain.py — chain expanding-window WF backtests year-over-year,
passing the prior year's ending balance into the next year's --balance.

Mirrors the locked WF run identically (Dec 20 + witching + HFOIV p70x0.25 r6 lb90
+ tiered Kelly + -10% kill switch). The only knob is which strategy variant
to chain (full tiers vs half tiers vs anything else with the same per-year
filename pattern).
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STRAT = os.path.join(_ROOT, "bot", "strategies")

# (label, start, end, strategy_filename)
def build_legs(suffix: str, end_2026: str):
    s = suffix  # "" for full tiers, "-half" for half tiers
    return [
        ("2021",     "20210101", "20211231", f"wf-2020-test-2021-slotbl-non3{s}.json"),
        ("2022",     "20220101", "20221231", f"wf-2020-2021-test-2022-slotbl-non3{s}.json"),
        ("2023",     "20230101", "20231231", f"wf-2020-2022-test-2023-slotbl-non3{s}.json"),
        ("2024",     "20240101", "20241231", f"wf-2020-2023-test-2024-slotbl-non3{s}.json"),
        ("2025",     "20250101", "20251231", f"mixed-best-ev-wf-2020-2024-slotbl-non3{s}.json"),
        ("2026 YTD", "20260101", end_2026,   f"mixed-best-ev-wf-2020-2025-slotbl-non3{s}.json"),
    ]

def run_year(strategy_path: str, start: str, end: str, balance: float) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        out = f.name
    cmd = [
        "python3", os.path.join(_ROOT, "bot", "backtest", "backtester.py"),
        "--strategy-file", strategy_path,
        "--start", start, "--end", end,
        "--risk-tiers",
        "--hfoiv", "--hfoiv-rolling", "6", "--hfoiv-lookback", "90",
        "--balance", f"{balance:.2f}", "--risk-pct", "0.01",
        "--json-output", out,
    ]
    print(f"[run] {start}->{end} bal=${balance:,.0f} strat={os.path.basename(strategy_path)}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout[-2000:])
        raise SystemExit(f"backtester failed for {start}->{end}")
    with open(out) as f:
        results = json.load(f)
    os.unlink(out)
    return results["summary"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suffix", default="-half",
                   help='strategy suffix: "" for full tiers, "-half" for half tiers')
    p.add_argument("--start-balance", type=float, default=80000.0)
    p.add_argument("--end-2026", default="20260407",
                   help="end date for 2026 YTD leg (default: 20260407)")
    args = p.parse_args()

    legs = build_legs(args.suffix, args.end_2026)

    bal = args.start_balance
    rows = []
    for label, s, e, fname in legs:
        sp = os.path.join(_STRAT, fname)
        if not os.path.exists(sp):
            raise SystemExit(f"missing strategy: {sp}")
        summary = run_year(sp, s, e, bal)
        net = summary["net_pnl"]
        new_bal = bal + net
        rows.append({
            "year": label, "start": bal, "end": new_bal,
            "pct": (new_bal / bal - 1) * 100,
            "pf": summary.get("profit_factor"),
            "dd": summary.get("max_dd_pct"),
            "trades": summary.get("total_trades"),
        })
        bal = new_bal

    print()
    label = "FULL tiers" if args.suffix == "" else f'tiers="{args.suffix}"'
    print(f"=== Compounded chain ({label}) ===")
    print(f"{'year':<10}{'start':>12}{'end':>12}{'pct':>10}{'PF':>8}{'maxDD':>8}{'trades':>8}")
    for r in rows:
        pf = f"{r['pf']:.2f}" if r['pf'] else "—"
        print(f"{r['year']:<10}${r['start']:>10,.0f}${r['end']:>10,.0f}"
              f"{r['pct']:>+9.1f}%{pf:>8}{r['dd']:>7.1f}%{r['trades']:>8}")
    total_pct = (rows[-1]["end"] / args.start_balance - 1) * 100
    mult = rows[-1]["end"] / args.start_balance
    print(f"\n${args.start_balance:,.0f} -> ${rows[-1]['end']:,.0f}  "
          f"({total_pct:+.1f}% / {mult:.2f}x)")

if __name__ == "__main__":
    main()
