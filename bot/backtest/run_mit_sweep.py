"""Run MIT entry slippage sweep across all years."""
import sys, os, json
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from bot.backtest.backtester import (
    load_1s_bars, load_databento_bars, load_strategy, run_backtest,
)

def run_all_years(mit_ticks, confirmed_limit=False, tp_mode="fixed"):
    strategy = load_strategy(strategy_id='mixed-best-ev-v3-slip-moderate')
    config = {
        'balance': 76000, 'risk_pct': 0.01, 'max_concurrent': 3,
        'max_daily_trades': 15, 'min_fvg_size': 0.25,
        'slip': False,
        'confirmed_limit': confirmed_limit,
        'mit_entry_ticks': mit_ticks,
        'risk_tiers': True,
        'tp_mode': tp_mode,
    }

    results = []
    # Databento years
    for year in range(2021, 2026):
        df = load_databento_bars('NQ', f'{year}0101', f'{year}1231')
        trades, final = run_backtest(df, strategy, config)
        wins = sum(1 for t in trades if t.is_win)
        losses = sum(1 for t in trades if t.exit_reason == 'SL')
        eod = sum(1 for t in trades if t.exit_reason == 'EOD')
        gross_p = sum(t.pnl_dollars for t in trades if t.pnl_dollars > 0)
        gross_l = sum(t.pnl_dollars for t in trades if t.pnl_dollars < 0)
        pf = abs(gross_p / gross_l) if gross_l else 0
        total_pnl = final - 76000

        # Equity curve for max DD
        bal = 76000; peak = bal; max_dd = 0; max_dd_pct = 0
        for t in trades:
            bal += t.pnl_dollars
            if bal > peak: peak = bal
            dd = peak - bal
            dd_pct = dd / peak if peak > 0 else 0
            if dd > max_dd: max_dd = dd
            if dd_pct > max_dd_pct: max_dd_pct = dd_pct

        results.append({
            'year': year, 'trades': len(trades), 'wins': wins,
            'wr': round(wins/len(trades)*100, 1) if trades else 0,
            'pnl': round(total_pnl), 'ret': round(total_pnl/76000*100, 1),
            'pf': round(pf, 2), 'max_dd_pct': round(max_dd_pct*100, 1),
        })

    # IB 1s 2026
    df = load_1s_bars(os.path.join(os.path.dirname(__file__), '..', 'data'), '20260102', '20260325')
    trades, final = run_backtest(df, strategy, config)
    wins = sum(1 for t in trades if t.is_win)
    gross_p = sum(t.pnl_dollars for t in trades if t.pnl_dollars > 0)
    gross_l = sum(t.pnl_dollars for t in trades if t.pnl_dollars < 0)
    pf = abs(gross_p / gross_l) if gross_l else 0
    total_pnl = final - 76000
    bal = 76000; peak = bal; max_dd = 0; max_dd_pct = 0
    for t in trades:
        bal += t.pnl_dollars
        if bal > peak: peak = bal
        dd = peak - bal
        dd_pct = dd / peak if peak > 0 else 0
        if dd > max_dd: max_dd = dd
        if dd_pct > max_dd_pct: max_dd_pct = dd_pct
    results.append({
        'year': '2026Q1', 'trades': len(trades), 'wins': wins,
        'wr': round(wins/len(trades)*100, 1) if trades else 0,
        'pnl': round(total_pnl), 'ret': round(total_pnl/76000*100, 1),
        'pf': round(pf, 2), 'max_dd_pct': round(max_dd_pct*100, 1),
    })

    # Print summary
    if confirmed_limit:
        label = "CONFIRMED ENTRY + TP TOUCH  |  Entry 1 tick past (fill at limit), TP touch = fill"
    else:
        label = f"MIT ENTRY = {mit_ticks} TICK(S)  |  TP = fill on touch"
    tp_label = {
        "fixed": "Fixed TP",
        "runner": "Runner (SL stays)",
        "runner-be": "Runner (SL→BE+1)",
        "split": "Split ((n-1) TP + 1 runner)",
    }[tp_mode]
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  {tp_label}  |  Risk tiers ON")
    print(f"{'='*70}")
    print(f"  {'Year':<8} {'Trades':>7} {'WR%':>6} {'Net P&L':>12} {'Return':>8} {'PF':>6} {'MaxDD':>7}")
    print(f"  {'-'*8} {'-'*7} {'-'*6} {'-'*12} {'-'*8} {'-'*6} {'-'*7}")
    total_pnl_all = 0
    total_trades_all = 0
    for r in results:
        total_pnl_all += r['pnl']
        total_trades_all += r['trades']
        print(f"  {str(r['year']):<8} {r['trades']:>7} {r['wr']:>5.1f}% ${r['pnl']:>10,} {r['ret']:>7.1f}% {r['pf']:>5.2f} {r['max_dd_pct']:>6.1f}%")
    print(f"  {'-'*8} {'-'*7} {'-'*6} {'-'*12} {'-'*8} {'-'*6} {'-'*7}")
    print(f"  {'TOTAL':<8} {total_trades_all:>7} {'':>6} ${total_pnl_all:>10,}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    # Usage: python run_mit_sweep.py <mode> [tp_mode]
    # mode: 1-4 (MIT ticks), confirmed
    # tp_mode: fixed (default), runner, runner-be, split
    mode = sys.argv[1] if len(sys.argv) > 1 else '1'
    tp_mode = sys.argv[2] if len(sys.argv) > 2 else 'fixed'
    if mode == 'confirmed':
        run_all_years(0, confirmed_limit=True, tp_mode=tp_mode)
    else:
        run_all_years(int(mode), tp_mode=tp_mode)
