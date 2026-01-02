"""
Overview/Dashboard Principal - Fraud Analytics
"""
import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils import load_data, get_summary_stats, get_temporal_data, get_country_stats

dash.register_page(__name__, path='/', name="Dashboard", order=0)

DARK_STYLE = {
    "background-color": "#000000",
    "color": "#FFFFFF",
    "font-family": "'Inter', sans-serif",
}

# Load data
df = load_data("df_exp_50_2.csv")
stats = get_summary_stats(df)
temporal_data = get_temporal_data(df, freq='H') if df is not None else None
country_stats = get_country_stats(df) if df is not None else None

def create_country_fraud_chart(country_stats):
    if country_stats is None or country_stats.empty:
        return go.Figure()

    cs = country_stats.head(10).copy()

    fig = px.bar(
        cs,
        x="country",
        y="fraud_rate_smoothed_pct",
        color="fraud_rate_smoothed_pct",
        color_continuous_scale="Reds",
        hover_data={
            "total": True,
            "fraud_count": True,
            "fraud_rate_pct": ":.2f",
            "fraud_rate_smoothed_pct": ":.2f",
        },
        title=""
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter", "color": "white"},
        xaxis_title="Country",
        yaxis_title="Fraud Rate (smoothed, %)",
        margin=dict(l=20, r=20, t=20, b=40),
        showlegend=False,
    )

    fig.update_xaxes(tickangle=-30)

    return fig
fig_countries = create_country_fraud_chart(country_stats)


# Create KPI Cards
def create_kpi_card(title, value, subtitle="", color="#00d4ff", icon=""):
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.H6(title, className="text-muted mb-2", style={"font-size": "0.85rem"}),
                html.H2(f"{value:,.0f}" if isinstance(value, (int, float)) else str(value), 
                       style={"color": color, "font-weight": "700", "margin": "0"}),
                html.P(subtitle, className="text-muted mt-2 mb-0", style={"font-size": "0.75rem"}) if subtitle else None
            ])
        ])
    ], style={"background": "#111", "border": "1px solid #333", "border-radius": "10px", "height": "100%"})

# Create temporal chart
def create_temporal_chart():
    if temporal_data is None or temporal_data.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    # Total transactions
    fig.add_trace(go.Scatter(
        x=temporal_data['timestamp'],
        y=temporal_data['count'],
        name='Total Transactions',
        line=dict(color='#00d4ff', width=2),
        fill='tonexty',
        fillcolor='rgba(0, 212, 255, 0.1)'
    ))
    
    # Fraud transactions
    fig.add_trace(go.Scatter(
        x=temporal_data['timestamp'],
        y=temporal_data['fraud_count'],
        name='Fraudulent',
        line=dict(color='#ff0055', width=2),
        fill='tonexty',
        fillcolor='rgba(255, 0, 85, 0.1)'
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter', 'color': 'white'},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode='x unified'
    )
    
    return fig

# Create fraud rate over time
def create_fraud_rate_chart():
    if temporal_data is None or temporal_data.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=temporal_data['timestamp'],
        y=temporal_data['fraud_rate'] * 100,
        name='Fraud Rate (%)',
        line=dict(color='#ff0055', width=3),
        fill='tozeroy',
        fillcolor='rgba(255, 0, 85, 0.2)'
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter', 'color': 'white'},
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis_title="Fraud Rate (%)",
        hovermode='x unified'
    )
    
    return fig

# Amount distribution
def create_amount_distribution():
    if df is None or 'Amount' not in df.columns:
        return go.Figure()
    
    fig = go.Figure()
    
    legit = df[df['Class']==0]['Amount'] if 'Class' in df.columns else df['Amount']
    fraud = df[df['Class']==1]['Amount'] if 'Class' in df.columns else pd.Series()
    
    if len(legit) > 0:
        fig.add_trace(go.Histogram(
            x=legit,
            name='Legitimate',
            nbinsx=50,
            marker_color='#00d4ff',
            opacity=0.7
        ))
    
    if len(fraud) > 0:
        fig.add_trace(go.Histogram(
            x=fraud,
            name='Fraudulent',
            nbinsx=50,
            marker_color='#ff0055',
            opacity=0.7
        ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter', 'color': 'white'},
        barmode='overlay',
        xaxis_title="Transaction Amount",
        yaxis_title="Frequency",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

layout = dbc.Container(fluid=True, style=DARK_STYLE, children=[
    html.Link(href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700&display=swap", rel="stylesheet"),
    
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Fraud Analytics Dashboard", className="my-4", style={"font-weight": "700"}),
            html.P("Real-time monitoring and analysis of fraudulent transactions", className="text-muted mb-4")
        ], width=12)
    ]),
    
    # KPI Cards Row
    dbc.Row([
        dbc.Col(create_kpi_card(
            "Total Transactions",
            stats.get('total_transactions', 0),
            f"Last updated: {datetime.now().strftime('%H:%M')}"
        ), width=12, lg=3, className="mb-4"),
        
        dbc.Col(create_kpi_card(
            "Fraudulent Transactions",
            stats.get('fraud_count', 0),
            f"{stats.get('fraud_rate', 0):.2f}% of total",
            color="#ff0055"
        ), width=12, lg=3, className="mb-4"),
        
        dbc.Col(create_kpi_card(
            "Total Amount",
            f"${stats.get('total_amount', 0):,.0f}",
            f"Avg: ${stats.get('avg_amount', 0):,.2f}"
        ), width=12, lg=3, className="mb-4"),
        
        dbc.Col(create_kpi_card(
            "Fraud Amount",
            f"${stats.get('fraud_amount', 0):,.0f}",
            "Potential loss prevented",
            color="#ff0055"
        ), width=12, lg=3, className="mb-4"),
    ], className="g-4"),
    
    # Charts Row 1
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Transaction Volume Over Time", className="text-info"),
                dbc.CardBody([
                    dcc.Graph(
                        figure=create_temporal_chart(),
                        config={'displayModeBar': False}
                    )
                ])
            ], style={"background": "#111", "border": "1px solid #333"})
        ], width=12, lg=8, className="mb-4"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Fraud Rate Trend", className="text-danger"),
                dbc.CardBody([
                    dcc.Graph(
                        figure=create_fraud_rate_chart(),
                        config={'displayModeBar': False}
                    )
                ])
            ], style={"background": "#111", "border": "1px solid #333"})
        ], width=12, lg=4, className="mb-4"),
    ], className="g-4"),
    
    # Charts Row 2
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Transaction Amount Distribution", className="text-info"),
                dbc.CardBody([
                    dcc.Graph(
                        figure=create_amount_distribution(),
                        config={'displayModeBar': False}
                    )
                ])
            ], style={"background": "#111", "border": "1px solid #333"})
        ], width=12, lg=6, className="mb-4"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Top Countries by Fraud Rate", className="text-danger"),
                dbc.CardBody([
                    dcc.Graph(
                        figure=fig_countries,
                        config={'displayModeBar': False}
                    )
                ])
            ], style={"background": "#111", "border": "1px solid #333"})
        ], width=12, lg=6, className="mb-4"),
    ], className="g-4"),
])

