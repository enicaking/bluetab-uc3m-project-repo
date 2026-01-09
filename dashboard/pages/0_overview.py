"""
Overview / Main Dashboard Page - Fraud Analytics

This page is an "overview" of the dataset using historical (ground-truth) fraud labels.
It does NOT run real-time predictions here. Model performance and ROI are shown in other pages.

Fixes included:
1) Dropdowns readable in dark mode (done via assets/styles.css).
2) Replaced the slider (which looked like a double bar) with a numeric input (cleaner).
3) KPI numbers never break into multiple lines (nowrap class).
4) Removed ugly/unhelpful hover popups (hover disabled).
5) Fraud rate chart redesigned: clean smoothed trend (+ optional raw faint line).
6) Log-scale histogram ticks improved (10, 100, 1k, 10k...).
7) Unified colors through constants.
"""

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# Allow importing utils.py from dashboard/../
sys.path.append(str(Path(__file__).parent.parent))
from utils import load_data, get_summary_stats, get_temporal_data, get_country_stats

dash.register_page(__name__, path="/", name="Dashboard", order=0)

# Import design system
try:
    from design_system import (
        DARK_STYLE, CARD_STYLE, PANEL_STYLE, KPI_CARD_STYLE,
        ACCENT_PRIMARY, ACCENT_DANGER, ACCENT_SUCCESS,
        get_gradient_text_style, get_accent_color
    )
    ACCENT_TEAL = ACCENT_PRIMARY
    ACCENT_PINK = ACCENT_DANGER
except ImportError:
    # Fallback if design_system not available
    BG = "#0a0a0a"
    PANEL_BG = "#141414"
    BORDER = "#2a2a2a"
    TEXT = "#FFFFFF"
    ACCENT_TEAL = "#00d4ff"
    ACCENT_PINK = "#ff0055"
    DARK_STYLE = {"backgroundColor": BG, "color": TEXT, "fontFamily": "'Inter', sans-serif"}
    CARD_STYLE = {"background": PANEL_BG, "border": f"1px solid {BORDER}", "borderRadius": "12px", "height": "100%"}
    PANEL_STYLE = {"background": PANEL_BG, "border": f"1px solid {BORDER}"}
    KPI_CARD_STYLE = CARD_STYLE
    ACCENT_PRIMARY = ACCENT_TEAL
    ACCENT_DANGER = ACCENT_PINK
    ACCENT_SUCCESS = "#00ff88"
    def get_gradient_text_style(*args, **kwargs): return {}
    def get_accent_color(*args, **kwargs): return ACCENT_TEAL

# -----------------------------
# Load data once
# -----------------------------
df = load_data("df_exp_same_prop.csv")  # TODO: escojer el dataset que queramos


# -----------------------------
# Helpers
# -----------------------------
def filter_df_by_split(df_in: pd.DataFrame, split: str) -> pd.DataFrame:
    """
    Filter the dataset by Train/Test/All, because the dataset covers ~2 unique calendar days.
    
    Note: Even though preprocessing.ipynb separates train/test before oversampling,
    the final exported CSV (df_exp_*) contains all data concatenated with original timestamps.
    This function allows users to filter by day in the dashboard UI.
    
    - Train = earliest day in timestamps (Day 1)
    - Test  = second day (Day 2)
    - All   = everything (both days)
    """
    if df_in is None or df_in.empty or "timestamp" not in df_in.columns:
        return df_in

    dff = df_in[df_in["timestamp"].notna()].copy()
    if dff.empty:
        return dff

    days = sorted(dff["timestamp"].dt.date.unique())
    if len(days) == 0:
        return dff

    if split == "all" or len(days) == 1:
        return dff

    train_day = days[0]
    test_day = days[1] if len(days) > 1 else days[0]

    if split == "train":
        return dff[dff["timestamp"].dt.date == train_day]
    if split == "test":
        return dff[dff["timestamp"].dt.date == test_day]

    return dff


def _base_layout(fig: go.Figure) -> go.Figure:
    """Apply consistent dark styling to all figures."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter", "color": "white"},
        margin=dict(l=20, r=20, t=25, b=20),
        hovermode=False,  # <- disables hover box globally (clean)
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_kpi_card(title: str, value_id: str, subtitle_id: str, color: str):
    """
    KPI card with gradient styling, just a card with a title, a value and a subtitle.
    """
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.H6(title, className="kpi-label", style={"marginBottom": "0.5rem"}),
                        html.H2(
                            id=value_id, 
                            className="kpi-value", 
                            style={
                                "color": color, 
                                "fontWeight": "700", 
                                "margin": "0",
                                "background": f"linear-gradient(135deg, {color} 0%, {color}dd 100%)",
                                "-webkit-background-clip": "text",
                                "-webkit-text-fill-color": "transparent",
                                "background-clip": "text",
                            }
                        ),
                        html.P(id=subtitle_id, className="text-muted mt-2 mb-0", style={"fontSize": "0.75rem", "opacity": "0.8"}),
                    ],
                    style={"position": "relative", "zIndex": 1}
                ),
                # Decorative gradient line
                html.Div(
                    style={
                        "position": "absolute",
                        "top": 0,
                        "left": 0,
                        "right": 0,
                        "height": "3px",
                        "background": f"linear-gradient(90deg, {color} 0%, {color}88 100%)",
                        "opacity": 0.6,
                    }
                )
            ],
            style={"position": "relative", "padding": "1.5rem"}
        ),
        className="kpi-card",
        style={**KPI_CARD_STYLE, "height": "100%"},
    )

# GRAPHS:

def create_temporal_chart(temporal_data: pd.DataFrame) -> go.Figure:
    """Total vs fraud volume over time."""
    if temporal_data is None or temporal_data.empty:
        return go.Figure()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=temporal_data["timestamp"],
            y=temporal_data["count"],
            name="Total Transactions",
            line=dict(color=ACCENT_TEAL, width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=temporal_data["timestamp"],
            y=temporal_data["fraud_count"],
            name="Fraudulent",
            line=dict(color=ACCENT_PINK, width=2),
        )
    )

    fig.update_yaxes(title_text="Transactions (count)", rangemode="tozero")
    return _base_layout(fig)


def create_fraud_rate_chart(temporal_data: pd.DataFrame, rolling_window: int = 3, show_raw: bool = True) -> go.Figure:
    """
    Fraud rate trend:
    - main line = rolling mean (smooth)
    - optional raw line = faint (for reference)
    """
    if temporal_data is None or temporal_data.empty:
        return go.Figure()

    td = temporal_data.copy()

    # Ensure percent scale (0-100)
    if td["fraud_rate"].max() <= 1.0:
        td["fraud_rate"] = td["fraud_rate"] * 100

    td = td[td["count"] > 0].copy()

    td["fraud_rate_smooth"] = td["fraud_rate"].rolling(
        window=rolling_window, min_periods=1, center=True
    ).mean()

    fig = go.Figure()

    if show_raw:
        fig.add_trace(
            go.Scatter(
                x=td["timestamp"],
                y=td["fraud_rate"],
                name="Raw",
                line=dict(color=ACCENT_TEAL, width=1),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=td["timestamp"],
            y=td["fraud_rate_smooth"],
            name=f"Smoothed (rolling {rolling_window})",
            line=dict(color=ACCENT_PINK, width=3),
        )
    )

    # Auto-scale based on actual data range (no fixed 0-100)
    min_rate = td["fraud_rate_smooth"].min()
    max_rate = td["fraud_rate_smooth"].max()
    # Add some padding (10% on each side)
    padding = (max_rate - min_rate) * 0.1 if max_rate > min_rate else max_rate * 0.1
    y_min = max(0, min_rate - padding)
    y_max = max_rate + padding
    
    fig.update_yaxes(
        title_text="Fraud Rate (%)", 
        range=[y_min, y_max], 
        ticksuffix="%",
        rangemode="normal"
    )
    return _base_layout(fig)


def create_amount_distribution(dff: pd.DataFrame) -> go.Figure:
    """
    Amount histogram with log-scale Y.
    Fraudulent transactions are more visible with better styling.
    """
    if dff is None or dff.empty or "Amount" not in dff.columns or "Class" not in dff.columns:
        return go.Figure()

    legit = dff.loc[dff["Class"] == 0, "Amount"]
    fraud = dff.loc[dff["Class"] == 1, "Amount"]

    fig = go.Figure()

    # Legitimate: lighter, lower opacity
    fig.add_trace(go.Histogram(
        x=legit, 
        name="Legitimate", 
        nbinsx=50, 
        opacity=0.4, 
        marker_color=ACCENT_TEAL,
        marker_line_width=0
    ))
    
    # Fraudulent: more visible, higher opacity, with border
    fig.add_trace(go.Histogram(
        x=fraud, 
        name="Fraudulent", 
        nbinsx=50, 
        opacity=0.9, 
        marker_color=ACCENT_PINK,
        marker_line_color="#ff3366",
        marker_line_width=1
    ))

    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text="Transaction Amount")
    fig.update_yaxes(
        title_text="Transaction count (log10)",
        type="log",
        tickmode="array",
        tickvals=[1, 10, 100, 1000, 10000, 100000],
        ticktext=["1", "10", "100", "1k", "10k", "100k"],
    )

    return _base_layout(fig)


def create_country_chart(country_stats: pd.DataFrame, rank_by: str = "rate") -> go.Figure:
    """
    Country ranking with color gradient based on percentage:
    - rank_by="rate": top by smoothed fraud rate (%)
    - rank_by="amount": top by fraud_amount (impact)
    
    Colors: darker red = higher fraud rate/amount
    """
    if country_stats is None or country_stats.empty:
        return go.Figure()

    cs = country_stats.copy()

    if rank_by == "amount":
        if "fraud_amount" not in cs.columns:
            fig = go.Figure()
            fig.add_annotation(
                text="fraud_amount not available. Add it in utils.get_country_stats().",
                x=0.5, y=0.5, showarrow=False, font=dict(color="white")
            )
            return _base_layout(fig)

        cs = cs.sort_values("fraud_amount", ascending=False).head(10)
        y_col = "fraud_amount"
        y_title = "Fraud Amount (impact)"
        color_col = "fraud_amount"
    else:
        cs = cs.sort_values("fraud_rate_smoothed_pct", ascending=False).head(10)
        y_col = "fraud_rate_smoothed_pct"
        y_title = "Fraud Rate (smoothed, %)"
        color_col = "fraud_rate_smoothed_pct"

    # Create color gradient based on values
    min_val = cs[color_col].min()
    max_val = cs[color_col].max()
    val_range = max_val - min_val if max_val > min_val else 1
    
    # Normalize to 0-1 for color mapping
    cs['color_norm'] = (cs[color_col] - min_val) / val_range
    
    # Create colors: from light pink to dark red
    colors = []
    for norm in cs['color_norm']:
        # Interpolate between light pink (#ff6699) and dark red (#cc0000)
        r = int(255 - (norm * 51))  # 255 -> 204
        g = int(102 - (norm * 102))  # 102 -> 0
        b = int(153 - (norm * 51))   # 153 -> 102
        colors.append(f"rgb({r}, {g}, {b})")

    fig = go.Figure()
    
    # Prepare hover text with detailed information
    hover_texts = []
    for idx, row in cs.iterrows():
        hover_parts = [f"<b>{row['country']}</b>"]
        if rank_by == "rate":
            hover_parts.append(f"Fraud Rate: {row[y_col]:.2f}%")
            hover_parts.append(f"Raw Rate: {row.get('fraud_rate_pct', 0):.2f}%")
        else:
            hover_parts.append(f"Fraud Amount: ${row[y_col]:,.0f}")
        hover_parts.append(f"Total Transactions: {int(row['total']):,}")
        hover_parts.append(f"Fraud Count: {int(row['fraud_count']):,}")
        if 'total_amount' in row and pd.notna(row['total_amount']):
            hover_parts.append(f"Total Amount: ${row['total_amount']:,.0f}")
        hover_texts.append("<br>".join(hover_parts))
    
    fig.add_trace(go.Bar(
        x=cs["country"],
        y=cs[y_col],
        marker=dict(
            color=colors,
            line=dict(color="#ff0055", width=1)
        ),
        text=[f"{val:.1f}%" if rank_by == "rate" else f"${val:,.0f}" for val in cs[y_col]],
        textposition="outside",
        hovertemplate="%{hovertext}<extra></extra>",
        hovertext=hover_texts,
    ))

    fig.update_xaxes(title_text="Country", tickangle=-25)
    fig.update_yaxes(title_text=y_title, rangemode="tozero")

    if rank_by == "rate":
        fig.update_yaxes(ticksuffix="%")

    # Apply base layout but override hovermode for this chart
    fig = _base_layout(fig)
    fig.update_layout(
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="rgba(20, 20, 20, 0.95)",
            bordercolor="#00d4ff",
            font_size=12,
            font_family="Inter",
            font_color="white",
        )
    )
    
    return fig


# -----------------------------
# Layout
# -----------------------------
layout = dbc.Container(
    fluid=True,
    style=DARK_STYLE,
    children=[
        html.Link(
            href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700&display=swap",
            rel="stylesheet",
        ),

        # Title Section
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.H1(
                                    "Fraud Analytics Dashboard", 
                                    className="my-4",
                                    style={
                                        **get_gradient_text_style(),
                                        "fontWeight": "700",
                                        "fontSize": "2.5rem",
                                        "letterSpacing": "-1px",
                                    }
                                ),
                                html.P(
                                    "Real-time monitoring and analysis of fraudulent transactions",
                                    className="text-muted",
                                    style={"fontSize": "0.95rem", "marginTop": "-10px", "opacity": "0.8"},
                                ),
                            ],
                            className="section-header"
                        )
                    ],
                    width=12,
                )
            ]
        ),

        # Controls 
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Dataset split", className="text-muted", style={"fontSize": "0.8rem"}),
                        dcc.Dropdown(
                            id="ov-split",
                            options=[
                                {"label": "Train (Day 1)", "value": "train"},
                                {"label": "Test (Day 2)", "value": "test"},
                                {"label": "All (Train + Test)", "value": "all"},
                            ],
                            value="test",
                            clearable=False,
                        ),
                    ],
                    width=12,
                    lg=3,
                    className="mb-3",
                ),

                dbc.Col(
                    [
                        html.Label("Country ranking", className="text-muted", style={"fontSize": "0.8rem"}),
                        dcc.Dropdown(
                            id="ov-country-rankby",
                            options=[
                                {"label": "Top by fraud rate (smoothed)", "value": "rate"},
                                {"label": "Top by fraud amount (impact)", "value": "amount"},
                            ],
                            value="rate",
                            clearable=False,
                        ),
                    ],
                    width=12,
                    lg=3,
                    className="mb-3",
                ),

                # Replace slider with a numeric input (clean + no “double bar” issue)
                dbc.Col(
                    [
                        html.Label("Min transactions per country", className="text-muted", style={"fontSize": "0.8rem"}),
                        dbc.Input(
                            id="ov-min-tx",
                            type="number",
                            min=10,
                            max=2000,
                            step=10,
                            value=100,
                        ),
                    ],
                    width=12,
                    lg=3,
                    className="mb-3",
                ),
            ],
            className="g-3",
        ),

        # KPI row
        dbc.Row(
            [
                dbc.Col(create_kpi_card("Total Transactions", "ov-kpi-total", "ov-kpi-subtitle", ACCENT_TEAL), width=12, lg=3, className="mb-4"),
                dbc.Col(create_kpi_card("Fraudulent Transactions", "ov-kpi-fraud", "ov-kpi-fraud-subtitle", ACCENT_PINK), width=12, lg=3, className="mb-4"),
                dbc.Col(create_kpi_card("Total Transaction Amount", "ov-kpi-total-amt", "ov-kpi-total-amt-sub", ACCENT_TEAL), width=12, lg=3, className="mb-4"),
                dbc.Col(create_kpi_card("Observed Fraud Amount", "ov-kpi-fraud-amt", "ov-kpi-fraud-amt-sub", ACCENT_PINK), width=12, lg=3, className="mb-4"),
            ],
            className="g-4",
        ),

        # Charts row 1
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                [
                                    html.Span("📊 ", style={"marginRight": "0.5rem"}),
                                    "Transaction Volume Over Time"
                                ],
                                className="text-info",
                                style={"fontWeight": "600", "letterSpacing": "0.3px"}
                            ),
                            dbc.CardBody([dcc.Graph(id="ov-volume", config={"displayModeBar": False})]),
                        ],
                        className="card",
                        style=PANEL_STYLE,
                    ),
                    width=12,
                    lg=8,
                    className="mb-4",
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                [
                                    html.Span("📈 ", style={"marginRight": "0.5rem"}),
                                    "Fraud Rate Trend"
                                ],
                                className="text-danger",
                                style={"fontWeight": "600", "letterSpacing": "0.3px"}
                            ),
                            dbc.CardBody([dcc.Graph(id="ov-fraud-rate", config={"displayModeBar": False})]),
                        ],
                        className="card",
                        style=PANEL_STYLE,
                    ),
                    width=12,
                    lg=4,
                    className="mb-4",
                ),
            ],
            className="g-4",
        ),

        # Charts row 2
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Transaction Amount Distribution (log count)", className="text-info"),
                            dbc.CardBody([dcc.Graph(id="ov-amount", config={"displayModeBar": False})]),
                        ],
                        style=PANEL_STYLE,
                    ),
                    width=12,
                    lg=6,
                    className="mb-4",
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Top Countries", className="text-danger"),
                            dbc.CardBody([dcc.Graph(id="ov-countries", config={"displayModeBar": False})]),
                        ],
                        style=PANEL_STYLE,
                    ),
                    width=12,
                    lg=6,
                    className="mb-4",
                ),
            ],
            className="g-4",
        ),
    ],
)


# -----------------------------
# Callback: update KPIs + charts
# -----------------------------
@callback(
    Output("ov-kpi-total", "children"),
    Output("ov-kpi-subtitle", "children"),
    Output("ov-kpi-fraud", "children"),
    Output("ov-kpi-fraud-subtitle", "children"),
    Output("ov-kpi-total-amt", "children"),
    Output("ov-kpi-total-amt-sub", "children"),
    Output("ov-kpi-fraud-amt", "children"),
    Output("ov-kpi-fraud-amt-sub", "children"),
    Output("ov-volume", "figure"),
    Output("ov-fraud-rate", "figure"),
    Output("ov-amount", "figure"),
    Output("ov-countries", "figure"),
    Input("ov-split", "value"),
    Input("ov-country-rankby", "value"),
    Input("ov-min-tx", "value"),
)
def update_overview(split, rank_by, min_tx):
    if df is None or df.empty:
        empty = go.Figure()
        return "0", "", "0", "", "$0", "", "$0", "", empty, empty, empty, empty

    # Filter by split (train/test/all)
    dff = filter_df_by_split(df, split)

    # KPIs
    stats_f = get_summary_stats(dff)

    # Format numbers to ensure they're complete and readable
    total_tx = stats_f.get('total_transactions', 0)
    kpi_total = f"{total_tx:,}" if total_tx >= 1000 else str(total_tx)
    kpi_total_sub = f"Split: {split.upper()} • Updated: {datetime.now().strftime('%H:%M')}"

    fraud_count = stats_f.get('fraud_count', 0)
    kpi_fraud = f"{fraud_count:,}" if fraud_count >= 1000 else str(fraud_count)
    kpi_fraud_sub = f"{stats_f.get('fraud_rate', 0):.2f}% of total"

    total_amt = stats_f.get('total_amount', 0)
    if total_amt >= 1e9:
        kpi_total_amt = f"${total_amt/1e9:.2f}B"
    elif total_amt >= 1e6:
        kpi_total_amt = f"${total_amt/1e6:.2f}M"
    elif total_amt >= 1e3:
        kpi_total_amt = f"${total_amt/1e3:.2f}K"
    else:
        kpi_total_amt = f"${total_amt:,.0f}"
    kpi_total_amt_sub = f"Avg per tx: ${stats_f.get('avg_amount', 0):,.2f}"

    # Important: this is NOT prevented loss, it's observed fraud volume
    fraud_amt = stats_f.get('fraud_amount', 0)
    if fraud_amt >= 1e9:
        kpi_fraud_amt = f"${fraud_amt/1e9:.2f}B"
    elif fraud_amt >= 1e6:
        kpi_fraud_amt = f"${fraud_amt/1e6:.2f}M"
    elif fraud_amt >= 1e3:
        kpi_fraud_amt = f"${fraud_amt/1e3:.2f}K"
    else:
        kpi_fraud_amt = f"${fraud_amt:,.0f}"
    kpi_fraud_amt_sub = "Observed (ground-truth) fraud amount"

    # Time aggregation (always hourly)
    temporal = get_temporal_data(dff, freq="H")

    # Smoothing window for hourly data
    rolling_window = 3

    fig_vol = create_temporal_chart(temporal)
    fig_rate = create_fraud_rate_chart(temporal, rolling_window=rolling_window, show_raw=False)
    fig_amount = create_amount_distribution(dff)

    # Country stats
    min_tx = int(min_tx) if min_tx is not None else 100
    cs = get_country_stats(dff, min_tx=min_tx)
    fig_countries = create_country_chart(cs, rank_by=rank_by)

    return (
        kpi_total, kpi_total_sub,
        kpi_fraud, kpi_fraud_sub,
        kpi_total_amt, kpi_total_amt_sub,
        kpi_fraud_amt, kpi_fraud_amt_sub,
        fig_vol, fig_rate, fig_amount, fig_countries
    )
