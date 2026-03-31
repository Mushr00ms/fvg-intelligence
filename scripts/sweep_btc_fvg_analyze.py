#!/usr/bin/env python3
"""
sweep_btc_fvg_analyze.py — Post-sweep analysis: re-bucket risk_bps, build strategy matrices.

Reads per-trade JSON from sweep_btc_fvg.py and:
  1. Applies various risk_bps bucket schemes (uniform, log-spaced, quantile)
  2. Builds (time_period × risk_bucket × setup) → EV matrices
  3. Ranks configs by aggregate quality metrics
  4. Identifies cells that pass strategy-grade filters (200+ samples, positive EV)

Usage:
    python3 scripts/sweep_btc_fvg_analyze.py
    python3 scripts/sweep_btc_fvg_analyze.py --trades-file scripts/btc_sweep_results/sweep_trades.json
    python3 scripts/sweep_btc_fvg_analyze.py --top 5 --min-samples 100
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

N_VALUES = [round(1.0 + i * 0.25, 2) for i in range(9)]


def _is_win(val):
    """Handle both bool and string outcomes from JSON."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val == "True"
    return bool(val)


# ── Risk bucket schemes ──────────────────────────────────────────────────

def uniform_buckets(min_bps, max_bps, n_buckets):
    """Generate uniform-width bucket edges."""
    step = (max_bps - min_bps) / n_buckets
    return [round(min_bps + i * step, 1) for i in range(n_buckets + 1)]


def log_buckets(min_bps, max_bps, n_buckets):
    """Generate log-spaced bucket edges (denser at small values)."""
    log_min = math.log(max(min_bps, 1))
    log_max = math.log(max_bps)
    step = (log_max - log_min) / n_buckets
    return [round(math.exp(log_min + i * step), 1) for i in range(n_buckets + 1)]


def quantile_buckets(risk_values, n_buckets):
    """Generate bucket edges from data quantiles."""
    import numpy as np
    if len(risk_values) < n_buckets:
        return None
    percentiles = np.linspace(0, 100, n_buckets + 1)
    edges = [round(float(np.percentile(risk_values, p)), 1) for p in percentiles]
    # Deduplicate
    edges = sorted(set(edges))
    return edges if len(edges) >= 3 else None


# ── Cell computation ─────────────────────────────────────────────────────

def compute_cells(trades, bucket_edges, min_samples=50):
    """Build (time_period, risk_bucket, setup) → stats from trade records.

    Args:
        trades: List of trade dicts (from sweep_trades.json)
        bucket_edges: List of float edges for risk_bps bucketing
        min_samples: Minimum trades per cell

    Returns:
        List of cell dicts with EV/WR at each R:R level.
    """
    # Index trades into buckets
    cells = defaultdict(list)

    for t in trades:
        risk_bps = t["risk_bps"]
        time_period = t["time_period"]
        setup = t["setup"]

        # Find bucket
        bucket_label = None
        for i in range(len(bucket_edges) - 1):
            if bucket_edges[i] <= risk_bps < bucket_edges[i + 1]:
                bucket_label = f"{bucket_edges[i]:.0f}-{bucket_edges[i+1]:.0f}"
                break

        if bucket_label is None:
            continue

        cells[(time_period, bucket_label, setup)].append(t)

    # Compute stats per cell
    result = []
    for (tp, rb, setup), cell_trades in cells.items():
        n = len(cell_trades)
        if n < min_samples:
            continue

        rr_stats = {}
        for nv in N_VALUES:
            nv_str = str(nv)
            wins = sum(1 for t in cell_trades if _is_win(t["outcomes"].get(nv_str, False)))
            wr = wins / n
            ev = wr * (nv + 1) - 1
            rr_stats[nv] = {
                "wins": wins,
                "total": n,
                "win_rate": round(wr, 4),
                "ev": round(ev, 4),
            }

        best_n = max(rr_stats, key=lambda k: rr_stats[k]["ev"])
        best_ev = rr_stats[best_n]["ev"]

        avg_risk_bps = sum(t["risk_bps"] for t in cell_trades) / n

        result.append({
            "time_period": tp,
            "risk_range": rb,
            "setup": setup,
            "samples": n,
            "avg_risk_bps": round(avg_risk_bps, 1),
            "best_n": best_n,
            "best_ev": best_ev,
            "rr_stats": rr_stats,
        })

    return result


def score_bucket_scheme(cells, min_ev=0.0):
    """Score a bucket scheme by how many strategy-grade cells it produces.

    Returns dict with:
        total_cells: number of cells with enough samples
        positive_ev_cells: cells where best_ev > min_ev
        avg_best_ev: average best EV across positive cells
        total_samples: sum of samples in positive cells
        coverage_score: positive_ev_cells * avg_best_ev (composite)
    """
    if not cells:
        return {"total_cells": 0, "positive_ev_cells": 0, "avg_best_ev": 0,
                "total_samples": 0, "coverage_score": 0}

    positive = [c for c in cells if c["best_ev"] > min_ev]
    avg_ev = sum(c["best_ev"] for c in positive) / len(positive) if positive else 0
    total_samples = sum(c["samples"] for c in positive)

    return {
        "total_cells": len(cells),
        "positive_ev_cells": len(positive),
        "avg_best_ev": round(avg_ev, 4),
        "total_samples": total_samples,
        "coverage_score": round(len(positive) * avg_ev, 4),
    }


# ── Analysis ─────────────────────────────────────────────────────────────

def analyze_config(label, trades, min_samples=50):
    """Analyze a single config's trades with multiple bucket schemes.

    Returns dict with per-scheme results + overall ranking metrics.
    """
    if not trades:
        return {"label": label, "total_trades": 0, "schemes": {}}

    import numpy as np
    risk_values = [t["risk_bps"] for t in trades]
    r_min, r_max = min(risk_values), max(risk_values)
    r_p5, r_p95 = np.percentile(risk_values, [5, 95])

    # Trim extremes for bucket construction
    trimmed_min = max(r_p5, 1)
    trimmed_max = r_p95

    schemes = {}

    # Uniform schemes
    for n_buckets in [5, 7, 10]:
        name = f"uniform_{n_buckets}"
        edges = uniform_buckets(trimmed_min, trimmed_max, n_buckets)
        cells = compute_cells(trades, edges, min_samples)
        schemes[name] = {
            "edges": edges,
            "cells": cells,
            "score": score_bucket_scheme(cells),
        }

    # Log schemes
    for n_buckets in [5, 7, 10]:
        name = f"log_{n_buckets}"
        edges = log_buckets(trimmed_min, trimmed_max, n_buckets)
        cells = compute_cells(trades, edges, min_samples)
        schemes[name] = {
            "edges": edges,
            "cells": cells,
            "score": score_bucket_scheme(cells),
        }

    # Quantile schemes
    for n_buckets in [5, 7, 10]:
        name = f"quantile_{n_buckets}"
        edges = quantile_buckets(risk_values, n_buckets)
        if edges:
            cells = compute_cells(trades, edges, min_samples)
            schemes[name] = {
                "edges": edges,
                "cells": cells,
                "score": score_bucket_scheme(cells),
            }

    # Find best scheme
    best_scheme = max(schemes.items(), key=lambda kv: kv[1]["score"]["coverage_score"])

    return {
        "label": label,
        "total_trades": len(trades),
        "risk_bps_range": (round(r_min, 1), round(r_max, 1)),
        "risk_bps_p5_p95": (round(r_p5, 1), round(r_p95, 1)),
        "best_scheme": best_scheme[0],
        "best_score": best_scheme[1]["score"],
        "schemes": schemes,
    }


# ── Display ──────────────────────────────────────────────────────────────

def print_strategy_matrix(cells, title=""):
    """Print a readable strategy matrix from cells."""
    if not cells:
        print(f"  {title}: no qualifying cells")
        return

    # Sort by time_period then risk_range
    cells = sorted(cells, key=lambda c: (c["time_period"], c["risk_range"]))

    print(f"\n{'='*100}")
    if title:
        print(f"  {title}")
    print(f"{'='*100}")

    header = (
        f"{'Time Period':<14} {'Risk (bps)':<12} {'Setup':<14} {'Samples':>7} "
        f"{'Best N':>6} {'EV':>7} {'WR@1.5':>7} {'WR@2.0':>7} {'WR@2.5':>7}"
    )
    print(header)
    print("-" * 100)

    for c in cells:
        rr = c["rr_stats"]
        wr_15 = rr.get(1.5, {}).get("win_rate", 0) * 100
        wr_20 = rr.get(2.0, {}).get("win_rate", 0) * 100
        wr_25 = rr.get(2.5, {}).get("win_rate", 0) * 100

        ev_marker = "+" if c["best_ev"] > 0 else " "
        print(
            f"{c['time_period']:<14} {c['risk_range']:<12} {c['setup']:<14} "
            f"{c['samples']:>7} {c['best_n']:>6.2f} {ev_marker}{c['best_ev']:>6.4f} "
            f"{wr_15:>6.1f}% {wr_20:>6.1f}% {wr_25:>6.1f}%"
        )

    # Summary
    positive = [c for c in cells if c["best_ev"] > 0]
    print(f"\n  Total cells: {len(cells)}, Positive EV: {len(positive)}, "
          f"Total samples: {sum(c['samples'] for c in cells):,}")


def print_top_configs(analyses, limit=10):
    """Print the top configs ranked by coverage score."""
    ranked = sorted(analyses, key=lambda a: a.get("best_score", {}).get("coverage_score", 0),
                    reverse=True)

    print(f"\n{'='*100}")
    print("  TOP CONFIGS BY COVERAGE SCORE")
    print(f"{'='*100}")

    header = (
        f"{'#':>3} {'Label':<50} {'Trades':>7} {'Scheme':<15} "
        f"{'Cells':>5} {'+EV':>4} {'Avg EV':>7} {'Score':>7}"
    )
    print(header)
    print("-" * 100)

    for i, a in enumerate(ranked[:limit]):
        sc = a.get("best_score", {})
        print(
            f"{i+1:>3} {a['label']:<50} {a['total_trades']:>7} "
            f"{a.get('best_scheme', ''):<15} {sc.get('total_cells', 0):>5} "
            f"{sc.get('positive_ev_cells', 0):>4} {sc.get('avg_best_ev', 0):>7.4f} "
            f"{sc.get('coverage_score', 0):>7.4f}"
        )

    print("=" * 100)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze BTC FVG sweep results")
    parser.add_argument("--trades-file",
                        default=os.path.join(_ROOT, "scripts", "btc_sweep_results", "sweep_trades.json"))
    parser.add_argument("--summary-file",
                        default=os.path.join(_ROOT, "scripts", "btc_sweep_results", "sweep_summary.json"))
    parser.add_argument("--top", type=int, default=10, help="Number of top configs to show detail")
    parser.add_argument("--min-samples", type=int, default=50, help="Min samples per cell")
    parser.add_argument("--output",
                        default=os.path.join(_ROOT, "scripts", "btc_sweep_results", "sweep_analysis.json"))
    args = parser.parse_args()

    # Load trades
    if os.path.exists(args.trades_file):
        print(f"Loading trades from {args.trades_file}...")
        with open(args.trades_file) as f:
            all_trades = json.load(f)
        print(f"  {len(all_trades)} configs with trade data")
    else:
        print(f"ERROR: No trades file at {args.trades_file}")
        print("  Run sweep_btc_fvg.py with --save-trades first")
        sys.exit(1)

    # Analyze each config
    print(f"\nAnalyzing with min_samples={args.min_samples}...")
    analyses = []
    for label, trades in all_trades.items():
        if not trades:
            continue
        analysis = analyze_config(label, trades, args.min_samples)
        analyses.append(analysis)

    # Print top configs
    print_top_configs(analyses, limit=args.top)

    # Print detailed matrix for the best config
    ranked = sorted(analyses, key=lambda a: a.get("best_score", {}).get("coverage_score", 0),
                    reverse=True)

    for i, best in enumerate(ranked[:args.top]):
        scheme_name = best.get("best_scheme", "")
        scheme_data = best.get("schemes", {}).get(scheme_name, {})
        cells = scheme_data.get("cells", [])

        # Only positive EV cells
        positive_cells = [c for c in cells if c["best_ev"] > 0]
        print_strategy_matrix(
            positive_cells,
            title=f"#{i+1} {best['label']} — scheme={scheme_name}, "
                  f"edges={scheme_data.get('edges', [])}"
        )

    # Save analysis
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Strip cells from output (too large), keep scores + edges
    save_data = []
    for a in analyses:
        entry = {k: v for k, v in a.items() if k != "schemes"}
        entry["scheme_scores"] = {
            name: {"edges": s["edges"], "score": s["score"]}
            for name, s in a.get("schemes", {}).items()
        }
        save_data.append(entry)

    with open(args.output, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nAnalysis saved to {args.output}")

    # Also save best strategy matrix as standalone
    if ranked:
        best = ranked[0]
        scheme_name = best.get("best_scheme", "")
        scheme_data = best.get("schemes", {}).get(scheme_name, {})
        positive_cells = [c for c in scheme_data.get("cells", []) if c["best_ev"] > 0]

        matrix_path = args.output.replace(".json", "_best_matrix.json")
        matrix_data = {
            "config": best["label"],
            "scheme": scheme_name,
            "bucket_edges": scheme_data.get("edges", []),
            "score": best.get("best_score", {}),
            "cells": positive_cells,
        }
        with open(matrix_path, "w") as f:
            json.dump(matrix_data, f, indent=2, default=str)
        print(f"Best strategy matrix saved to {matrix_path}")


if __name__ == "__main__":
    main()
