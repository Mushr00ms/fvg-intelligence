"""
replay_runner.py — CLI for Databento tick-accurate replay.

Modes:
  --plan       Analyze backtest results and show recommended replay dates
  --compare-only  Compare existing replay results with backtest (no replay)
  (default)    Run tick-accurate replay from Databento trades data
"""

import argparse
import os
import sys

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _run_plan(args):
    """Analyze backtest and print replay plan."""
    from bot.replay.scenario_selector import load_and_select, print_replay_plan

    if not args.backtest:
        print("Error: --backtest is required for --plan mode")
        sys.exit(1)

    selected = load_and_select(
        args.backtest,
        max_per_scenario=args.max_per_scenario,
        max_total=args.max_total,
    )
    print_replay_plan(selected)


def _run_compare(args):
    """Compare existing replay results with backtest."""
    from bot.replay.comparator import compare_directory

    if not args.backtest:
        print("Error: --backtest is required for --compare-only mode")
        sys.exit(1)

    replay_dir = args.output_dir or os.path.join(os.path.dirname(__file__), "results")
    if os.path.isdir(replay_dir) and os.listdir(replay_dir):
        compare_directory(replay_dir, args.backtest)
    else:
        print(f"No replay results found in {replay_dir}")


def _run_replay(args):
    """Run tick-accurate replay from Databento trades data."""
    from bot.bot_config import BotConfig
    from bot.replay.tick_data_feed import TickDataFeed, find_trades_zips, DEFAULT_EXTRACT_DIR
    from bot.replay.replay_engine import ReplayEngine

    # Find Databento zip files
    zip_paths = []
    if args.zip:
        zip_paths = [args.zip]
    else:
        zip_paths = find_trades_zips(_ROOT)
        if not zip_paths:
            print("Error: No Databento trades zip found. Use --zip to specify.")
            sys.exit(1)
    print(f"Trades zips: {[os.path.basename(z) for z in zip_paths]}")

    extract_dir = args.extract_dir or DEFAULT_EXTRACT_DIR

    # Build config
    config = BotConfig(
        replay_mode=True,
        strategy_dir=args.strategy_dir or "",
        use_risk_tiers=args.risk_tiers,
        margin_fallback_per_contract=args.margin,
    )

    # Set balance on config for the engine
    config._replay_start_balance = args.balance

    # Initialize data feed
    feed = TickDataFeed(zip_paths=zip_paths, extract_dir=extract_dir)
    print(f"Extracting trades to {extract_dir} ...")
    feed.prepare(verbose=True)

    # Output directory for results
    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), "results")

    # Run replay
    engine = ReplayEngine(config, feed, output_dir=output_dir)
    engine.run(args.start, args.end)

    # Compare with backtest if requested
    if args.compare:
        from bot.replay.comparator import compare_directory
        compare_directory(output_dir, args.compare)


def main():
    parser = argparse.ArgumentParser(
        description="Tick-accurate replay using Databento trades data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show replay plan (which dates to test):
  python -m bot.replay --plan --backtest bot/backtest/results/2025.json

  # Run full-year replay:
  python -m bot.replay --start 2025-01-01 --end 2025-12-31 --balance 100000

  # Run replay for specific scenario dates and compare:
  python -m bot.replay --start 2025-01-02 --end 2025-01-08 --balance 100000 \\
      --compare bot/backtest/results/2025.json

  # Compare already-collected results:
  python -m bot.replay --compare-only --backtest bot/backtest/results/2025.json
        """,
    )

    parser.add_argument("--plan", action="store_true",
                        help="Analyze backtest and show recommended replay dates")
    parser.add_argument("--compare-only", action="store_true",
                        help="Compare existing replay results (no replay)")
    parser.add_argument("--backtest", type=str, default=None,
                        help="Path to backtest results JSON")
    parser.add_argument("--compare", type=str, default=None,
                        help="Path to backtest JSON for post-replay comparison")
    parser.add_argument("--zip", type=str, default=None,
                        help="Path to Databento trades zip (auto-detected if omitted)")
    parser.add_argument("--extract-dir", type=str, default=None,
                        help="Directory for extracted .dbn.zst files (default: /tmp/databento_trades_nq)")
    parser.add_argument("--strategy-dir", type=str, default=None,
                        help="Path to strategy directory")
    parser.add_argument("--start", type=str, default="2025-01-01",
                        help="Start date YYYY-MM-DD (default: 2025-01-01)")
    parser.add_argument("--end", type=str, default="2025-12-31",
                        help="End date YYYY-MM-DD (default: 2025-12-31)")
    parser.add_argument("--balance", type=float, default=100000.0,
                        help="Starting balance (default: 100000)")
    parser.add_argument("--risk-tiers", action="store_true",
                        help="Enable 3-tier risk sizing")
    parser.add_argument("--margin", type=float, default=33000.0,
                        help="Margin per contract (default: 33000)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for replay results")
    parser.add_argument("--max-per-scenario", type=int, default=3,
                        help="Max dates per scenario for --plan (default: 3)")
    parser.add_argument("--max-total", type=int, default=30,
                        help="Max total dates for --plan (default: 30)")

    args = parser.parse_args()

    if args.plan:
        _run_plan(args)
    elif args.compare_only:
        _run_compare(args)
    else:
        _run_replay(args)


if __name__ == "__main__":
    main()
