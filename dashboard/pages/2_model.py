import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

dash.register_page(__name__, name="Model Metrics")

# --- Custom Styling ---
DARK_CARD = {"background-color": "#111", "border": "1px solid #333", "border-radius": "10px", "padding": "20px"}

# DATA PREPARATION 
models = ['Logistic Regression', 'Random Forest', 'XGBoost']
f1_scores = [0.82, 0.88, 0.91]
f2_scores = [0.85, 0.92, 0.95]

# Loading datasets for data synthesis comparison


# 1. Sleek Comparison Bar Chart
fig_metrics = go.Figure()
fig_metrics.add_trace(go.Bar(
    x=models, y=f1_scores, name='F1 Score',
    marker_color='#00d4ff', marker_line_color='#00d4ff', marker_line_width=1.5, opacity=0.8
))
fig_metrics.add_trace(go.Bar(
    x=models, y=f2_scores, name='F2 Score',
    marker_color='#ff0055', marker_line_color='#ff0055', marker_line_width=1.5, opacity=0.8
))

fig_metrics.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    barmode='group',
    font={'family': 'Inter'},
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=60, b=20)
)

# 2. Confusion Matrix (XGBoost Example)
# [True Negative, False Positive, False Negative, True Positive]
z = [[9500, 50], 
     [10, 440]] 
x = ['Predicted Legit', 'Predicted Fraud']
y = ['Actual Legit', 'Actual Fraud']

fig_cm = px.imshow(
    z, x=x, y=y, 
    color_continuous_scale='Reds',
    text_auto=True,
    title="Confusion Matrix (Best Model)"
)
fig_cm.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=20, r=20, t=60, b=20)
)

# --- Layout ---
layout = dbc.Container(fluid=True, children=[
    dbc.Row([
        dbc.Col(html.H2("Model Performance Analysis", className="mt-4 mb-2", style={"font-weight": "700"}), width=12),
        dbc.Col(html.P("Evaluating models with a focus on Recall (F2 Score) to minimize False Negatives.", className="text-muted mb-4"), width=12)
    ]),

    dbc.Row([
        # Metric Comparison Card
        dbc.Col(dbc.Card(style=DARK_CARD, children=[
            html.H5("Performance Summary", className="text-info"),
            dcc.Graph(figure=fig_metrics, config={'displayModeBar': False})
        ]), width=12, lg=7, className="mb-4"),

        # Confusion Matrix Card
        dbc.Col(dbc.Card(style=DARK_CARD, children=[
            html.H5("Error Analysis", className="text-danger"),
            dcc.Graph(figure=fig_cm, config={'displayModeBar': False})
        ]), width=12, lg=5, className="mb-4"),
    ]),

    # Summary Stats Row
    dbc.Row([
        dbc.Col(dbc.Alert("Best Performing Model: XGBoost", color="info", style={"text-align": "center"}), width=12)
    ])
])