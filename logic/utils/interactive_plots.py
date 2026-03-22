"""
Interactive Plotly visualizations for FVG analysis.

Generates interactive HTML heatmaps and optimizer charts
as companions to the existing static matplotlib PNGs.
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _inject_responsive_style(filepath):
    """Inject CSS, toggle button, and responsive layout into a Plotly HTML file."""
    head_inject = (
        '<style>'
        'html, body { margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; }'
        '#main-chart { width: 100% !important; height: 100vh !important; }'
        '.plotly-graph-div { width: 100% !important; height: 100% !important; }'
        '#toggle-labels { position: fixed; top: 10px; left: 10px; z-index: 9999;'
        '  padding: 6px 14px; border: 1px solid #555; border-radius: 6px;'
        '  background: rgba(30,30,30,0.85); color: #ddd; font-size: 12px;'
        '  cursor: pointer; font-family: sans-serif; }'
        '#toggle-labels:hover { background: rgba(60,60,60,0.95); }'
        '</style>'
    )
    body_inject = (
        '<button id="toggle-labels" onclick="toggleLabels()">Hide Labels</button>'
        '<script>'
        'var labelsVisible = true;'
        'function toggleLabels() {'
        '  var gd = document.getElementById("main-chart");'
        '  labelsVisible = !labelsVisible;'
        '  var annots = gd.layout.annotations || [];'
        '  var update = {};'
        '  for (var i = 0; i < annots.length; i++) {'
        '    if (annots[i].xref === "paper" || annots[i].yref === "paper") continue;'
        '    update["annotations[" + i + "].opacity"] = labelsVisible ? 1 : 0;'
        '  }'
        '  Plotly.relayout(gd, update);'
        '  document.getElementById("toggle-labels").textContent ='
        '    labelsVisible ? "Hide Labels" : "Show Labels";'
        '}'
        'window.addEventListener("load", function() {'
        '  var gd = document.getElementById("main-chart");'
        '  var upd = {};'
        '  var axes = ["yaxis","yaxis2","yaxis3","yaxis4","xaxis","xaxis2","xaxis3","xaxis4"];'
        '  axes.forEach(function(ax) {'
        '    if (gd.layout[ax]) upd[ax + ".automargin"] = true;'
        '  });'
        '  if (gd.layout.margin && gd.layout.margin.l < 100) upd["margin.l"] = 100;'
        '  var d = gd.data || [];'
        '  if (d.length === 4) {'
        '    Plotly.restyle(gd, {showscale: false}, [0, 2]);'
        '    Plotly.restyle(gd, {"colorbar.thickness": 15}, [1, 3]);'
        '  }'
        '  Plotly.relayout(gd, upd).then(function() { Plotly.Plots.resize(gd); });'
        '});'
        '</script>'
    )
    with open(filepath, 'r') as f:
        html = f.read()
    html = html.replace('<head>', f'<head>{head_inject}', 1)
    html = html.replace('</body>', f'{body_inject}</body>', 1)
    with open(filepath, 'w') as f:
        f.write(html)


def _sort_size_ranges(values):
    """Sort size range strings numerically (e.g. '0.25-1.25' by first number)."""
    def _key(s):
        try:
            return float(str(s).split("-")[0])
        except (ValueError, TypeError):
            return float("inf")
    return sorted(values, key=_key)


def _text_color_for_bg(colorscale_name, norm_value):
    """Return 'black' or 'white' based on perceived brightness of the colorscale at norm_value."""
    import plotly.colors as pc

    # Get the RGB values from the colorscale at the normalized position
    try:
        scale = pc.get_colorscale(colorscale_name)
    except Exception:
        return "white"

    # Find the two surrounding stops
    norm = max(0.0, min(1.0, norm_value))
    r, g, b = 128, 128, 128  # fallback
    for i in range(len(scale) - 1):
        low_pos, low_color = scale[i]
        high_pos, high_color = scale[i + 1]
        if low_pos <= norm <= high_pos:
            # Interpolate between the two stops
            t = (norm - low_pos) / (high_pos - low_pos) if high_pos != low_pos else 0
            # Parse rgb strings like "rgb(r,g,b)"
            def parse_rgb(c):
                if isinstance(c, str) and c.startswith("rgb"):
                    nums = c.replace("rgb(", "").replace(")", "").split(",")
                    return [int(x.strip()) for x in nums]
                # Try hex
                if isinstance(c, str) and c.startswith("#"):
                    c = c.lstrip("#")
                    return [int(c[i:i+2], 16) for i in (0, 2, 4)]
                return [128, 128, 128]
            lo = parse_rgb(low_color)
            hi = parse_rgb(high_color)
            r = int(lo[0] + t * (hi[0] - lo[0]))
            g = int(lo[1] + t * (hi[1] - lo[1]))
            b = int(lo[2] + t * (hi[2] - lo[2]))
            break

    # Perceived brightness (ITU-R BT.601)
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if brightness > 140 else "white"


def _sort_time_periods(values):
    """Sort time period strings chronologically (e.g. '09:30-10:00' by start time)."""
    def _key(s):
        try:
            start = str(s).split("-")[0].strip()
            h, m = start.split(":")
            return int(h) * 60 + int(m)
        except (ValueError, TypeError, IndexError):
            return 9999
    return sorted(values, key=_key)


def _build_pivot(df, value_col, aggfunc="mean"):
    """Build a pivot table with sorted rows (time) and columns (size)."""
    pivot = df.pivot_table(
        index="time_period",
        columns="size_range",
        values=value_col,
        aggfunc=aggfunc,
    )
    sorted_cols = _sort_size_ranges(pivot.columns)
    sorted_rows = _sort_time_periods(pivot.index)
    return pivot.loc[sorted_rows, sorted_cols]


def _make_hover_text(valid_df, pivot_shape, metric_name, pivot_values):
    """Build a hover text matrix showing all metrics per cell."""
    y_labels = list(pivot_values.index)
    x_labels = list(pivot_values.columns)
    hover = []
    for y in y_labels:
        row = []
        for x in x_labels:
            cell = valid_df[
                (valid_df["time_period"] == y) & (valid_df["size_range"] == x)
            ]
            if cell.empty:
                row.append("")
            else:
                c = cell.iloc[0]
                parts = [
                    f"<b>{metric_name}</b>: {pivot_values.loc[y, x]:.2f}" if pd.notna(pivot_values.loc[y, x]) else f"<b>{metric_name}</b>: N/A",
                    f"Size: {x}",
                    f"Time: {y}",
                    f"Total FVGs: {int(c.get('total_fvgs', 0))}",
                    f"Mitigation Rate: {c.get('mitigation_rate', 0):.1f}%",
                    f"Invalidation Rate: {c.get('invalidation_rate', 0):.1f}%",
                    f"Avg Expansion: {c.get('avg_expansion_size', float('nan')):.2f}" if pd.notna(c.get("avg_expansion_size")) else "Avg Expansion: N/A",
                ]
                if pd.notna(c.get("optimal_target")):
                    parts.append(f"Optimal Target: {c['optimal_target']:.1f}")
                    parts.append(f"Optimal EV: {c['optimal_ev']:.2f}")
                if pd.notna(c.get("avg_penetration_depth")):
                    parts.append(f"Avg Penetration Depth: {c['avg_penetration_depth']:.2f}")
                if pd.notna(c.get("avg_penetration_candle_count")):
                    parts.append(f"Avg Pen. Candles: {c['avg_penetration_candle_count']:.1f}")
                if pd.notna(c.get("avg_penetration_depth_ratio")):
                    parts.append(f"Pen. Depth Ratio: {c['avg_penetration_depth_ratio']:.1%}")
                row.append("<br>".join(parts))
        hover.append(row)
    return hover


def create_interactive_heatmaps(results_df, filename_prefix, save_dir="charts"):
    """
    Create a 4-panel interactive Plotly heatmap (HTML).

    Panels: Invalidation Rate, Avg Expansion Size, Total FVGs, Optimal Take-Profit Target.
    """
    os.makedirs(save_dir, exist_ok=True)
    valid = results_df[results_df["total_fvgs"] > 0].copy()
    if valid.empty:
        print("[WARNING] No valid results for interactive heatmaps")
        return None

    # Ensure size_range values are strings so Plotly handles categories consistently
    valid["size_range"] = valid["size_range"].astype(str)

    pivot_inv = _build_pivot(valid, "invalidation_rate")
    pivot_exp = _build_pivot(valid, "avg_expansion_size")
    pivot_count = _build_pivot(valid, "total_fvgs", aggfunc="sum")

    has_optimizer = "optimal_target" in valid.columns and valid["optimal_target"].notna().any()
    if has_optimizer:
        pivot_opt = _build_pivot(valid, "optimal_target")
    else:
        pivot_opt = _build_pivot(valid, "p75_expansion_size")

    fourth_title = "Optimal Take-Profit Target (pts)" if has_optimizer else "P75 Expansion Size (pts)"

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Invalidation Rate (%)",
            "Avg Expansion Size (pts)",
            "Total FVGs (count)",
            fourth_title,
        ],
        horizontal_spacing=0.12,
        vertical_spacing=0.12,
    )

    panels = [
        (pivot_inv, "RdYlGn_r", "Invalidation Rate (%)", ".1f", "%"),
        (pivot_exp, "YlGnBu", "Avg Expansion Size", ".1f", ""),
        (pivot_count, "Blues", "Total FVGs", "d", ""),
        (pivot_opt, "Viridis", fourth_title.split(" (")[0], ".1f", ""),
    ]

    for idx, (pivot, colorscale, metric_name, fmt, suffix) in enumerate(panels):
        row, col = divmod(idx, 2)
        row += 1
        col += 1

        hover = _make_hover_text(valid, pivot.shape, metric_name, pivot)

        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=[str(c) for c in pivot.columns],
                y=[str(r) for r in pivot.index],
                colorscale=colorscale,
                hovertext=hover,
                hoverinfo="text",
                colorbar=dict(
                    len=0.4,
                    y=1.0 - row * 0.5 + 0.25,
                    yanchor="middle",
                    thickness=15,
                ),
                showscale=(col == 2),
            ),
            row=row, col=col,
        )

        # Add text annotations with contrast color per cell
        zmin = np.nanmin(pivot.values) if not np.all(np.isnan(pivot.values)) else 0
        zmax = np.nanmax(pivot.values) if not np.all(np.isnan(pivot.values)) else 1
        zrange = zmax - zmin if zmax != zmin else 1

        # Determine which axis indices to use for this subplot
        ax_suffix = "" if idx == 0 else str(idx + 1)

        for i, y_val in enumerate(pivot.index):
            for j, x_val in enumerate(pivot.columns):
                val = pivot.iloc[i, j]
                if pd.isna(val):
                    continue
                if fmt == "d":
                    label = f"{int(val)}"
                else:
                    label = f"{val:{fmt}}{suffix}"
                norm = (val - zmin) / zrange
                txt_color = _text_color_for_bg(colorscale, norm)
                fig.add_annotation(
                    x=j, y=str(y_val), text=label,
                    showarrow=False,
                    font=dict(size=9, color=txt_color),
                    xref=f"x{ax_suffix}", yref=f"y{ax_suffix}",
                )

    # Force axis ordering: y = 09:30 at top, x = size ranges sorted numerically
    sorted_y = [str(r) for r in pivot_inv.index]
    sorted_x = [str(c) for c in pivot_inv.columns]
    for i in range(1, 5):
        ax_suffix = "" if i == 1 else str(i)
        fig.update_layout(**{
            f"yaxis{ax_suffix}": dict(
                categoryorder="array",
                categoryarray=sorted_y,
                autorange="reversed",
                automargin=True,
            ),
            f"xaxis{ax_suffix}": dict(
                categoryorder="array",
                categoryarray=sorted_x,
                type="category",
                range=[-0.5, len(sorted_x) - 0.5],
                automargin=True,
            ),
        })

    fig.update_layout(
        title_text=f"FVG Size-Time Analysis — {filename_prefix}",
        template="plotly_dark",
        autosize=True,
        margin=dict(l=100, r=60, t=80, b=60),
    )

    filepath = os.path.join(save_dir, f"{filename_prefix}_interactive.html")
    fig.write_html(
        filepath,
        include_plotlyjs='cdn',
        full_html=True,
        config={'responsive': True},
        div_id='main-chart',
    )
    _inject_responsive_style(filepath)
    print(f"[INFO] Interactive heatmap saved to {filepath}")
    return filepath


def create_optimizer_chart(curve_df, label, filename_prefix, save_dir="charts"):
    """
    Create a dual-axis EV optimization chart (HTML).

    Left axis: Expected Value curve.
    Right axis: Hit Rate curve.
    Vertical line at optimal target.
    """
    os.makedirs(save_dir, exist_ok=True)
    if curve_df.empty:
        print("[WARNING] Empty curve data — skipping optimizer chart")
        return None

    best_idx = curve_df["expected_value"].idxmax()
    best = curve_df.loc[best_idx]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=curve_df["target"],
            y=curve_df["expected_value"],
            name="Expected Value (T × P)",
            mode="lines",
            line=dict(color="#00cc96", width=2),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=curve_df["target"],
            y=curve_df["hit_rate"],
            name="Hit Rate P(reach T)",
            mode="lines",
            line=dict(color="#636efa", width=2, dash="dash"),
        ),
        secondary_y=True,
    )

    # Optimal target marker
    fig.add_trace(
        go.Scatter(
            x=[best["target"]],
            y=[best["expected_value"]],
            mode="markers+text",
            marker=dict(color="red", size=12, symbol="star"),
            text=[f"T*={best['target']:.1f}, EV={best['expected_value']:.2f}"],
            textposition="top center",
            name="Optimal Target",
        ),
        secondary_y=False,
    )

    # Vertical line at optimal
    fig.add_vline(
        x=best["target"],
        line_dash="dot",
        line_color="red",
        annotation_text=f"Optimal: {best['target']:.1f} pts",
        annotation_position="top right",
    )

    fig.update_layout(
        title_text=f"Expansion Target Optimizer — {label}",
        xaxis_title="Target (points)",
        template="plotly_dark",
        autosize=True,
        margin=dict(l=60, r=60, t=60, b=60),
    )
    fig.update_yaxes(title_text="Expected Value (pts)", secondary_y=False)
    fig.update_yaxes(title_text="Hit Rate", secondary_y=True)

    filepath = os.path.join(save_dir, f"{filename_prefix}_optimizer.html")
    fig.write_html(
        filepath,
        include_plotlyjs='cdn',
        full_html=True,
        config={'responsive': True},
        div_id='main-chart',
    )
    _inject_responsive_style(filepath)
    print(f"[INFO] Optimizer chart saved to {filepath}")
    return filepath


def create_interactive_mitigation_heatmap(results_df, filename_prefix, save_dir="charts"):
    """
    Interactive Plotly version of the mitigation time heatmap (HTML).
    """
    os.makedirs(save_dir, exist_ok=True)
    valid = results_df[results_df["total_fvgs"] > 0].copy()
    if valid.empty or "p75_mitigation_time" not in valid.columns:
        print("[WARNING] No valid data for interactive mitigation heatmap")
        return None

    pivot = _build_pivot(valid, "p75_mitigation_time")

    hover = []
    for y in pivot.index:
        row = []
        for x in pivot.columns:
            val = pivot.loc[y, x]
            cell = valid[(valid["time_period"] == y) & (valid["size_range"] == x)]
            if cell.empty or pd.isna(val):
                row.append("")
            else:
                c = cell.iloc[0]
                parts = [
                    f"<b>P75 Mitigation Time</b>: {val:.1f} min",
                    f"Size: {x}",
                    f"Time: {y}",
                    f"Total FVGs: {int(c.get('total_fvgs', 0))}",
                    f"Mitigated: {int(c.get('mitigated_fvgs', 0))}",
                ]
                row.append("<br>".join(parts))
        hover.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="YlOrRd",
            hovertext=hover,
            hoverinfo="text",
            colorbar=dict(title="Minutes"),
        )
    )

    # Add text annotations with contrast color
    zmin = np.nanmin(pivot.values) if not np.all(np.isnan(pivot.values)) else 0
    zmax = np.nanmax(pivot.values) if not np.all(np.isnan(pivot.values)) else 1
    zrange = zmax - zmin if zmax != zmin else 1
    for i, y in enumerate(pivot.index):
        for j, x in enumerate(pivot.columns):
            val = pivot.iloc[i, j]
            if pd.isna(val):
                continue
            norm = (val - zmin) / zrange
            txt_color = _text_color_for_bg("YlOrRd", norm)
            fig.add_annotation(
                x=x, y=y, text=f"{val:.0f}m",
                showarrow=False,
                font=dict(size=11, color=txt_color),
            )

    fig.update_layout(
        title_text=f"P75 Mitigation Time — {filename_prefix}",
        xaxis_title="FVG Size Range",
        yaxis_title="Time Period",
        template="plotly_dark",
        autosize=True,
        margin=dict(l=60, r=60, t=60, b=60),
        yaxis=dict(
            categoryorder="array",
            categoryarray=list(pivot.index),
            autorange="reversed",
        ),
        xaxis=dict(
            categoryorder="array",
            categoryarray=list(pivot.columns),
            type="category",
        ),
    )

    filepath = os.path.join(save_dir, f"{filename_prefix}_mitigation_interactive.html")
    fig.write_html(
        filepath,
        include_plotlyjs='cdn',
        full_html=True,
        config={'responsive': True},
        div_id='main-chart',
    )
    _inject_responsive_style(filepath)
    print(f"[INFO] Interactive mitigation heatmap saved to {filepath}")
    return filepath
