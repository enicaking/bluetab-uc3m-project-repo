"""
Data Exploration & EDA Page
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
sys.path.append(str(Path(__file__).parent.parent))
from utils import load_data, get_country_stats

dash.register_page(__name__, path='/data', name="Data Exploration", order=1)

# --- Styling Constants ---
DARK_STYLE = {
    "background-color": "#000000",
    "color": "#FFFFFF",
    "font-family": "'Inter', sans-serif",
}

DARK_CARD = {"background": "#111", "border": "1px solid #333", "border-radius": "10px", "padding": "20px"}

# Load data
df = load_data("df_exp_50_2.csv")

layout = dbc.Container(fluid=True, style=DARK_STYLE, children=[
    html.Link(href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700&display=swap", rel="stylesheet"),
    
    # Title
    dbc.Row([
        dbc.Col(html.H1("Exploratory Data Analysis", className="my-4", style={"font-weight": "700"}), width=12),
        dbc.Col(html.P("Comprehensive analysis of transaction patterns and fraud indicators", className="text-muted mb-4"), width=12)
    ]),
    
    # Dataset Selector
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Label("Select Dataset", className="text-muted mb-2"),
                    dcc.Dropdown(
                        id='dataset-selector',
                        options=[
                            {'label': 'Balanced 50/50', 'value': 'df_exp_50_2.csv'},
                            {'label': 'Balanced 63/37', 'value': 'df_exp_63_2.csv'},
                            {'label': 'Random Oversample', 'value': 'df_exp_random_2.csv'},
                            {'label': 'Same Proportion', 'value': 'df_exp_same_prop_2.csv'},
                        ],
                        value='df_exp_50_2.csv',
                        style={"color": "#000"}
                    )
                ])
            ], style=DARK_CARD)
        ], width=12, lg=4, className="mb-4")
    ]),
    
    # Summary Stats Cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Total Transactions", className="text-muted"),
                    html.H3(id="stat-total", style={"color": "#00d4ff", "font-weight": "700"})
                ])
            ], style=DARK_CARD)
        ], width=12, lg=3, className="mb-4"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Fraud Rate", className="text-muted"),
                    html.H3(id="stat-fraud-rate", style={"color": "#ff0055", "font-weight": "700"})
                ])
            ], style=DARK_CARD)
        ], width=12, lg=3, className="mb-4"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Avg Transaction Amount", className="text-muted"),
                    html.H3(id="stat-avg-amount", style={"color": "#00d4ff", "font-weight": "700"})
                ])
            ], style=DARK_CARD)
        ], width=12, lg=3, className="mb-4"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Features", className="text-muted"),
                    html.H3(id="stat-features", style={"color": "#00d4ff", "font-weight": "700"})
                ])
            ], style=DARK_CARD)
        ], width=12, lg=3, className="mb-4"),
    ], className="g-4"),
    
    # Charts Row 1
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Class Distribution", className="text-info"),
                dbc.CardBody([
                    dcc.Graph(id="class-distribution", config={'displayModeBar': False})
                ])
            ], style={"background": "#111", "border": "1px solid #333"})
        ], width=12, lg=6, className="mb-4"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Geographic Fraud Distribution", className="text-danger"),
                dbc.CardBody([
                    dcc.Graph(id="geo-map", config={'displayModeBar': False})
                ])
            ], style={"background": "#111", "border": "1px solid #333"})
        ], width=12, lg=6, className="mb-4"),
    ], className="g-4"),
    
    # Charts Row 2
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Transaction Amount by Class", className="text-info"),
                dbc.CardBody([
                    dcc.Graph(id="amount-boxplot", config={'displayModeBar': False})
                ])
            ], style={"background": "#111", "border": "1px solid #333"})
        ], width=12, lg=6, className="mb-4"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Time of Day Analysis", className="text-info"),
                dbc.CardBody([
                    dcc.Graph(id="time-analysis", config={'displayModeBar': False})
                ])
            ], style={"background": "#111", "border": "1px solid #333"})
        ], width=12, lg=6, className="mb-4"),
    ], className="g-4"),
    
    # Feature Analysis
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("PCA Features Correlation (Top 10)", className="text-info"),
                dbc.CardBody([
                    dcc.Graph(id="pca-correlation", config={'displayModeBar': False})
                ])
            ], style={"background": "#111", "border": "1px solid #333"})
        ], width=12, className="mb-4"),
    ]),
])

@callback(
    [Output("stat-total", "children"),
     Output("stat-fraud-rate", "children"),
     Output("stat-avg-amount", "children"),
     Output("stat-features", "children"),
     Output("class-distribution", "figure"),
     Output("geo-map", "figure"),
     Output("amount-boxplot", "figure"),
     Output("time-analysis", "figure"),
     Output("pca-correlation", "figure")],
    Input("dataset-selector", "value")
)
def update_data_exploration(dataset_name):
    from utils import load_data, get_country_stats
    
    df = load_data(dataset_name)
    if df is None or df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return "0", "0%", "$0", "0", empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
    
    # Stats
    total = len(df)
    fraud_rate = f"{df['Class'].mean() * 100:.2f}%" if 'Class' in df.columns else "0%"
    avg_amount = f"${df['Amount'].mean():,.2f}" if 'Amount' in df.columns else "$0"
    n_features = len([c for c in df.columns if c not in ['Class', 'transaction_id', 'customer_id', 'device_id', 'timestamp', 'date']])
    
    # Class Distribution
    if 'Class' in df.columns:
        class_counts = df['Class'].value_counts()
        fig_class = px.pie(
            values=class_counts.values,
            names=['Legitimate', 'Fraudulent'],
            color_discrete_map={'Legitimate': '#00d4ff', 'Fraudulent': '#ff0055'},
            title=""
        )
    else:
        fig_class = go.Figure()
    
    fig_class.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter', 'color': 'white'},
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    # Geographic Map
    country_stats = get_country_stats(df)
    if country_stats is not None and not country_stats.empty:
        fig_geo = px.choropleth(
            country_stats.head(20),
            locations="country",
            locationmode="country names",
            color="fraud_rate",
            hover_name="country",
            color_continuous_scale="Reds",
            title=""
        )
    else:
        fig_geo = go.Figure()
    
    fig_geo.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter', 'color': 'white'},
        geo=dict(bgcolor='rgba(0,0,0,0)', showlakes=True, lakecolor="#111", showcountries=True, countrycolor="#444"),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    # Amount Boxplot
    if 'Amount' in df.columns and 'Class' in df.columns:
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=df[df['Class']==0]['Amount'],
            name='Legitimate',
            marker_color='#00d4ff'
        ))
        fig_box.add_trace(go.Box(
            y=df[df['Class']==1]['Amount'],
            name='Fraudulent',
            marker_color='#ff0055'
        ))
    else:
        fig_box = go.Figure()
    
    fig_box.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter', 'color': 'white'},
        yaxis_title="Amount",
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    # Time Analysis
    if 'hour_sin' in df.columns and 'hour_cos' in df.columns and 'Class' in df.columns:
        # Convert back to hour
        df['hour'] = np.arctan2(df['hour_sin'], df['hour_cos']) * 12 / np.pi + 12
        df['hour'] = df['hour'] % 24
        
        time_fraud = df.groupby(df['hour'].round())['Class'].mean() * 100
        
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=time_fraud.index,
            y=time_fraud.values,
            mode='lines+markers',
            line=dict(color='#ff0055', width=3),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 85, 0.2)'
        ))
    else:
        fig_time = go.Figure()
    
    fig_time.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter', 'color': 'white'},
        xaxis_title="Hour of Day",
        yaxis_title="Fraud Rate (%)",
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    # PCA Correlation
    pca_cols = [f'V{i}' for i in range(1, 29) if f'V{i}' in df.columns]
    if len(pca_cols) >= 2:
        corr_matrix = df[pca_cols[:10]].corr()
        fig_corr = px.imshow(
            corr_matrix,
            color_continuous_scale='RdBu',
            aspect='auto',
            title=""
        )
    else:
        fig_corr = go.Figure()
    
    fig_corr.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter', 'color': 'white'},
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    return total, fraud_rate, avg_amount, n_features, fig_class, fig_geo, fig_box, fig_time, fig_corr
