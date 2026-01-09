"""
Data Exploration & EDA Page (Interactive)

- Uses real data from content via utils.load_data()
- Consistent styling with design_system + assets/styles.css
"""

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Allow importing utils.py from dashboard
sys.path.append(str(Path(__file__).parent.parent))
from utils import load_data, get_country_stats, get_temporal_data

dash.register_page(__name__, path="/data", name="Data Exploration", order=1)

# Design system 
try:
    from design_system import (
        DARK_STYLE, PANEL_STYLE, KPI_CARD_STYLE,
        ACCENT_PRIMARY, ACCENT_DANGER,
        get_gradient_text_style
    )
    ACCENT_TEAL = ACCENT_PRIMARY
    ACCENT_PINK = ACCENT_DANGER
except ImportError:
    BG = "#0a0a0a"
    PANEL_BG = "#141414"
    BORDER = "#2a2a2a"
    TEXT = "#FFFFFF"
    ACCENT_TEAL = "#00d4ff"
    ACCENT_PINK = "#ff0055"
    DARK_STYLE = {"backgroundColor": BG, "color": TEXT, "fontFamily": "'Inter', sans-serif", "minHeight": "100vh"}
    PANEL_STYLE = {"background": PANEL_BG, "border": f"1px solid {BORDER}", "borderRadius": "12px"}
    KPI_CARD_STYLE = PANEL_STYLE
    def get_gradient_text_style(*args, **kwargs): return {}

DATASET_OPTIONS = [
    {"label": "Balanced 50/50", "value": "df_exp_50.csv"},
    {"label": "Balanced 66/33", "value": "df_exp_63.csv"},
    {"label": "Random Oversample", "value": "df_exp_random.csv"},
    {"label": "Same Proportion", "value": "df_exp_same_prop.csv"},
]

DEFAULT_DATASET = "df_exp_50.csv"

# -----------------------------
# Helpers
# -----------------------------
def _base_layout(fig: go.Figure) -> go.Figure:
    """Apply consistent dark styling to all figures."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter", "color": "white"},
        margin=dict(l=20, r=20, t=10, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
    )
    return fig

def _empty_fig(message="No data available"):
    """Create an empty figure with a message."""
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, font=dict(color="white", size=14))
    return _base_layout(fig)

def _fmt_int(n: int) -> str:
    """Format integer with thousand separators."""
    return f"{int(n):,}"

def _fmt_money(x: float) -> str:
    """Format money with appropriate scale (K, M, B)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "$0"
    x = float(x)
    if abs(x) >= 1e9:
        return f"${x/1e9:.2f}B"
    if abs(x) >= 1e6:
        return f"${x/1e6:.2f}M"
    if abs(x) >= 1e3:
        return f"${x/1e3:.2f}K"
    return f"${x:,.2f}"

def create_kpi_card(title: str, value_id: str, color: str):
    """Create a KPI card with gradient styling."""
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
        style={**KPI_CARD_STYLE, "height": "100%", "zIndex": 1, "position": "relative"},
    )

def compute_feature_count(df: pd.DataFrame) -> int:
    """Count features excluding IDs and metadata."""
    exclude = {
        "Class", "transaction_id", "customer_id", "device_id",
        "timestamp", "date", "split", "is_synthetic"
    }
    cols = [c for c in df.columns if c not in exclude]
    return len(cols)


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

        # Title
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.H1(
                                    "Exploratory Data Analysis",
                                    className="my-4",
                                    style={
                                        **get_gradient_text_style(),
                                        "fontWeight": "700",
                                        "fontSize": "2.5rem",
                                        "letterSpacing": "-1px",
                                    },
                                ),
                                html.P(
                                    "Comprehensive analysis of transaction patterns and fraud indicators",
                                    className="text-muted",
                                    style={"fontSize": "0.95rem", "marginTop": "-10px", "opacity": "0.8"},
                                ),
                            ],
                            className="section-header",
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
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Label("Select Dataset", className="text-muted", style={"fontSize": "0.8rem", "marginBottom": "0.5rem"}),
                                html.Div(
                                    dcc.Dropdown(
                                        id="dataset-selector",
                                        options=DATASET_OPTIONS,
                                        value=DEFAULT_DATASET,
                                        clearable=False,
                                    ),
                                    style={"position": "relative", "zIndex": 10000}
                                ),
                            ]
                        ),
                        style={**PANEL_STYLE, "position": "relative", "zIndex": 10000},
                        className="card",
                    ),
                    width=12,
                    lg=4,
                    className="mb-4",
                )
            ],
            justify="center",
        ),

        # KPI row
        dbc.Row(
            [
                dbc.Col(create_kpi_card("Total Transactions", "stat-total", ACCENT_TEAL), width=12, lg=3, className="mb-4"),
                dbc.Col(create_kpi_card("Fraud Rate", "stat-fraud-rate", ACCENT_PINK), width=12, lg=3, className="mb-4"),
                dbc.Col(create_kpi_card("Avg Transaction Amount", "stat-avg-amount", ACCENT_TEAL), width=12, lg=3, className="mb-4"),
                dbc.Col(create_kpi_card("Features", "stat-features", ACCENT_PINK), width=12, lg=3, className="mb-4"),
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
                                    "Class Distribution"
                                ],
                                className="text-info",
                                style={"fontWeight": "600", "letterSpacing": "0.3px"}
                            ),
                            dbc.CardBody([dcc.Graph(id="class-distribution", config={"displayModeBar": False})]),
                        ],
                        style=PANEL_STYLE,
                        className="card",
                    ),
                    width=12,
                    lg=6,
                    className="mb-4",
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                [
                                    html.Span("🌍 ", style={"marginRight": "0.5rem"}),
                                    "Geographic Fraud Distribution"
                                ],
                                className="text-danger",
                                style={"fontWeight": "600", "letterSpacing": "0.3px"}
                            ),
                            dbc.CardBody([dcc.Graph(id="geo-map", config={"displayModeBar": False})]),
                        ],
                        style=PANEL_STYLE,
                        className="card",
                    ),
                    width=12,
                    lg=6,
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
                            dbc.CardHeader(
                                [
                                    html.Span("💰 ", style={"marginRight": "0.5rem"}),
                                    "Transaction Amount Distribution"
                                ],
                                className="text-info",
                                style={"fontWeight": "600", "letterSpacing": "0.3px"}
                            ),
                            dbc.CardBody([dcc.Graph(id="amount-distribution", config={"displayModeBar": False})]),
                        ],
                        style=PANEL_STYLE,
                        className="card",
                    ),
                    width=12,
                    lg=6,
                    className="mb-4",
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                [
                                    html.Span("⏰ ", style={"marginRight": "0.5rem"}),
                                    "Time of Day Analysis (Fraud Rate)"
                                ],
                                className="text-info",
                                style={"fontWeight": "600", "letterSpacing": "0.3px"}
                            ),
                            dbc.CardBody([dcc.Graph(id="time-analysis", config={"displayModeBar": False})]),
                        ],
                        style=PANEL_STYLE,
                        className="card",
                    ),
                    width=12,
                    lg=6,
                    className="mb-4",
                ),
            ],
            className="g-4",
        ),

        # Feature analysis
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                [
                                    html.Span("🔗 ", style={"marginRight": "0.5rem"}),
                                    "PCA Features Correlation Matrix"
                                ],
                                className="text-info",
                                style={"fontWeight": "600", "letterSpacing": "0.3px"}
                            ),
                            dbc.CardBody([dcc.Graph(id="pca-correlation", config={"displayModeBar": False})]),
                        ],
                        style=PANEL_STYLE,
                        className="card",
                    ),
                    width=12,
                    lg=6,
                    className="mb-4",
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                [
                                    html.Span("🎯 ", style={"marginRight": "0.5rem"}),
                                    "PCA Features Correlation with Class"
                                ],
                                className="text-danger",
                                style={"fontWeight": "600", "letterSpacing": "0.3px"}
                            ),
                            dbc.CardBody([dcc.Graph(id="pca-target-correlation", config={"displayModeBar": False})]),
                        ],
                        style=PANEL_STYLE,
                        className="card",
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
# Callback
# -----------------------------
@callback(
    Output("stat-total", "children"),
    Output("stat-fraud-rate", "children"),
    Output("stat-avg-amount", "children"),
    Output("stat-features", "children"),
    Output("class-distribution", "figure"),
    Output("geo-map", "figure"),
    Output("amount-distribution", "figure"),
    Output("time-analysis", "figure"),
    Output("pca-correlation", "figure"),
    Output("pca-target-correlation", "figure"),
    Input("dataset-selector", "value"),
)
def update_data_exploration(dataset_name: str):
    """Update all charts and KPIs based on selected dataset."""
    dff = load_data(dataset_name)
    if dff is None or dff.empty:
        ef = _empty_fig()
        return "0", "0.00%", "$0.00", "0", ef, ef, ef, ef, ef, ef

    # ---- KPIs
    total = _fmt_int(len(dff))
    fraud_rate_val = float(dff["Class"].mean() * 100) if "Class" in dff.columns else 0.0
    fraud_rate = f"{fraud_rate_val:.2f}%"
    avg_amount_val = float(dff["Amount"].mean()) if "Amount" in dff.columns else 0.0
    avg_amount = _fmt_money(avg_amount_val)
    n_features = str(compute_feature_count(dff))

    # ---- Class distribution
    fig_class = _empty_fig("Class column not found")
    if "Class" in dff.columns:
        counts = dff["Class"].value_counts().to_dict()
        legit = int(counts.get(0, 0))
        fraud = int(counts.get(1, 0))
        
        fig_class = px.pie(
            names=["Legitimate", "Fraudulent"],
            values=[legit, fraud],
            hole=0.0,
            color=["Legitimate", "Fraudulent"],
            color_discrete_map={"Legitimate": ACCENT_TEAL, "Fraudulent": ACCENT_PINK},
        )
        fig_class.update_traces(
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
        )
        fig_class = _base_layout(fig_class)

    # ---- Geographic map (use smoothed % + hover)
    fig_geo = _empty_fig("Country stats not available")
    cs = get_country_stats(dff, min_tx=100, m=500)
    if cs is not None and not cs.empty and "country" in cs.columns:
        # Prefer smoothed percent if available
        color_col = "fraud_rate_smoothed_pct" if "fraud_rate_smoothed_pct" in cs.columns else "fraud_rate_pct"
        if color_col not in cs.columns and "fraud_rate" in cs.columns:
            cs["fraud_rate_pct"] = cs["fraud_rate"] * 100
            color_col = "fraud_rate_pct"

        cs_plot = cs.sort_values(color_col, ascending=False).head(30).copy()

        fig_geo = px.choropleth(
            cs_plot,
            locations="country",
            locationmode="country names",
            color=color_col,
            hover_name="country",
            color_continuous_scale="Reds",
        )
        fig_geo.update_traces(
            hovertemplate=(
                "<b>%{location}</b><br>"
                "Fraud Rate: %{z:.2f}%<br>"
                "Total Transactions: %{customdata[0]:,}<br>"
                "Fraud Count: %{customdata[1]:,}<extra></extra>"
            ),
            customdata=cs_plot[["total", "fraud_count"]].values,
        )
        fig_geo.update_layout(
            coloraxis_colorbar=dict(title="Fraud Rate (%)"),
            geo=dict(
                bgcolor="rgba(0,0,0,0)",
                showcountries=True,
                countrycolor="#444",
                showocean=True,
                oceancolor="rgba(0,0,0,0)",
            ),
            margin=dict(l=0, r=0, t=0, b=0),
        )
        fig_geo = _base_layout(fig_geo)

    # ---- Amount distribution (histogram)
    fig_amount = _empty_fig("Amount/Class not available")
    if "Amount" in dff.columns and "Class" in dff.columns:
        legit_amt = dff.loc[dff["Class"] == 0, "Amount"]
        fraud_amt = dff.loc[dff["Class"] == 1, "Amount"]

        fig_amount = go.Figure()
        
        # Legitimate transactions: lighter, lower opacity
        fig_amount.add_trace(go.Histogram(
            x=legit_amt,
            name="Legitimate",
            nbinsx=50,
            opacity=0.4,
            marker_color=ACCENT_TEAL,
            marker_line_width=0,
            hovertemplate="<b>Legitimate</b><br>Amount: $%{x:,.2f}<br>Count: %{y:,}<extra></extra>",
        ))
        
        # Fraudulent transactions: more visible, higher opacity, with border
        fig_amount.add_trace(go.Histogram(
            x=fraud_amt,
            name="Fraudulent",
            nbinsx=50,
            opacity=0.9,
            marker_color=ACCENT_PINK,
            marker_line_color="#ff3366",
            marker_line_width=1,
            hovertemplate="<b>Fraudulent</b><br>Amount: $%{x:,.2f}<br>Count: %{y:,}<extra></extra>",
        ))

        fig_amount.update_layout(barmode="overlay")
        fig_amount.update_xaxes(title_text="Transaction Amount")
        fig_amount.update_yaxes(
            title_text="Transaction Count (log10)",
            type="log",
            tickmode="array",
            tickvals=[1, 10, 100, 1000, 10000, 100000],
            ticktext=["1", "10", "100", "1k", "10k", "100k"],
        )
        fig_amount = _base_layout(fig_amount)

    # ---- Time of day (fraud rate % by hour) using timestamps
    fig_time = _empty_fig("Timestamp not available")
    if "timestamp" in dff.columns and "Class" in dff.columns:
        tmp = dff[["timestamp", "Class"]].copy()
        tmp = tmp.loc[tmp["timestamp"].notna()].copy()
        
        if len(tmp) > 0:
            # Convert timestamp to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(tmp["timestamp"]):
                tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], errors="coerce")
            
            tmp = tmp.loc[tmp["timestamp"].notna()].copy()
            
            if len(tmp) > 0:
                # Extract hour from timestamp
                tmp["hour"] = tmp["timestamp"].dt.hour
                
                # Group by hour and calculate fraud rate
                hourly = tmp.groupby("hour")["Class"].agg(["mean", "count"]).reset_index()
                hourly.columns = ["hour", "fraud_rate", "count"]
                hourly["fraud_rate"] = hourly["fraud_rate"].astype(float) * 100
                
                # Ensure all 24 hours are present (fill missing with 0)
                hourly = hourly.set_index("hour").reindex(range(24), fill_value=0.0).reset_index()
                hourly["fraud_rate"] = hourly["fraud_rate"].fillna(0.0)
                hourly["count"] = hourly["count"].fillna(0.0)
                
                fig_time = go.Figure()
                fig_time.add_trace(
                    go.Scatter(
                        x=hourly["hour"],
                        y=hourly["fraud_rate"],
                        mode="lines+markers",
                        line=dict(color=ACCENT_PINK, width=3),
                        marker=dict(size=8, color=ACCENT_PINK),
                        fill="tozeroy",
                        fillcolor="rgba(255, 0, 85, 0.15)",
                        hovertemplate=(
                            "<b>Hour: %{x}:00</b><br>"
                            "Fraud Rate: %{y:.2f}%<br>"
                            "Transaction Count: %{customdata:,}<extra></extra>"
                        ),
                        customdata=hourly["count"].values,
                        name="Fraud rate",
                    )
                )
                fig_time.update_xaxes(title_text="Hour of Day", dtick=2, range=[-0.5, 23.5])
                fig_time.update_yaxes(title_text="Fraud Rate (%)", rangemode="tozero")
                fig_time = _base_layout(fig_time)
                fig_time.update_layout(
                    hoverlabel=dict(
                        bgcolor="rgba(20, 20, 20, 0.95)",
                        bordercolor=ACCENT_PINK,
                        font_size=12,
                        font_family="Inter",
                        font_color="white",
                    )
                )

    # ---- PCA correlation matrix (all available V-columns)
    fig_corr = _empty_fig("No PCA columns V1..V28 found")
    pca_cols = [f"V{i}" for i in range(1, 29) if f"V{i}" in dff.columns]
    if len(pca_cols) >= 2:
        # Use all available PCA columns, not just top 10
        pca_data = dff[pca_cols].dropna()
        if len(pca_data) > 0:
            corr = pca_data.corr()
            fig_corr = px.imshow(
                corr,
                color_continuous_scale="RdBu",
                zmin=-1, zmax=1,
                aspect="auto",
                labels=dict(color="Correlation"),
            )
            fig_corr.update_traces(
                hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>"
            )
            fig_corr.update_layout(
                coloraxis_colorbar=dict(title="Correlation"),
                xaxis_title="PCA Variable",
                yaxis_title="PCA Variable",
            )
            fig_corr = _base_layout(fig_corr)

    # ---- PCA correlation with Class target
    fig_target_corr = _empty_fig("No PCA columns or Class column found")
    if len(pca_cols) >= 1 and "Class" in dff.columns:
        pca_class_data = dff[pca_cols + ["Class"]].dropna()
        if len(pca_class_data) > 0:
            corr_with_target = pca_class_data[pca_cols + ["Class"]].corr()
            target_corr = corr_with_target["Class"].drop("Class")
            target_corr_abs = target_corr.abs().sort_values(ascending=False)
            target_corr_sorted = target_corr.reindex(target_corr_abs.index)
            
            # Create horizontal bar chart
            fig_target_corr = go.Figure()
            
            # Color bars: red for negative, teal for positive
            colors = [ACCENT_PINK if x < 0 else ACCENT_TEAL for x in target_corr_sorted.values]
            
            fig_target_corr.add_trace(
                go.Bar(
                    x=target_corr_sorted.values,
                    y=target_corr_sorted.index,
                    orientation="h",
                    marker=dict(color=colors),
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Correlation: %{x:.3f}<br>"
                        "|Correlation|: %{customdata:.3f}<extra></extra>"
                    ),
                    customdata=target_corr_abs.values,
                )
            )
            
            fig_target_corr.update_layout(
                xaxis_title="Correlation with Class",
                yaxis_title="PCA Variable",
                xaxis=dict(range=[-1, 1]),
                showlegend=False,
            )
            fig_target_corr.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.3)
            fig_target_corr = _base_layout(fig_target_corr)

    return total, fraud_rate, avg_amount, n_features, fig_class, fig_geo, fig_amount, fig_time, fig_corr, fig_target_corr
