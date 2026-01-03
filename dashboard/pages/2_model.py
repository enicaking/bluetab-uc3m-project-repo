"""
Model Performance & Metrics Page
"""
import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils import get_model_metrics

dash.register_page(__name__, path='/model', name="Model Performance", order=2)

# Import design system
try:
    from design_system import (
        DARK_STYLE, CARD_STYLE, PANEL_STYLE, KPI_CARD_STYLE,
        ACCENT_PRIMARY, ACCENT_DANGER, ACCENT_SUCCESS,
        get_gradient_text_style
    )
    DARK_CARD = KPI_CARD_STYLE
except ImportError:
    DARK_STYLE = {
        "background-color": "#0a0a0a",
        "color": "#FFFFFF",
        "font-family": "'Inter', sans-serif",
    }
    DARK_CARD = {"background": "#141414", "border": "1px solid #2a2a2a", "border-radius": "12px", "padding": "20px", "boxShadow": "0 4px 12px rgba(0,0,0,0.4)"}
    ACCENT_PRIMARY = "#00d4ff"
    ACCENT_DANGER = "#ff0055"
    ACCENT_SUCCESS = "#00ff88"
    def get_gradient_text_style(*args, **kwargs): return {}

# Get model metrics - will be loaded in callback to handle errors gracefully

layout = dbc.Container(fluid=True, style=DARK_STYLE, children=[
    html.Link(href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700&display=swap", rel="stylesheet"),
    
    # Header
    dbc.Row([
        dbc.Col([
            html.H1(
                "Model Performance Analysis", 
                className="my-4", 
                style={
                    **get_gradient_text_style(),
                    "font-weight": "700",
                    "fontSize": "2.5rem",
                    "letterSpacing": "-1px"
                }
            ),
            html.P(
                "Comprehensive evaluation of fraud detection models with focus on Recall (F2 Score)", 
                className="text-muted mb-4",
                style={"fontSize": "0.95rem", "opacity": "0.8"}
            )
        ], width=12)
    ]),
    
    # Model Comparison Cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Best Model", className="kpi-label"),
                    html.H3(
                        id="best-model-name", 
                        style={
                            "color": ACCENT_PRIMARY, 
                            "font-weight": "700",
                            "background": f"linear-gradient(135deg, {ACCENT_PRIMARY} 0%, {ACCENT_PRIMARY}dd 100%)",
                            "-webkit-background-clip": "text",
                            "-webkit-text-fill-color": "transparent",
                            "background-clip": "text",
                        }
                    ),
                    html.P(id="best-model-f2", className="text-success mt-2 mb-0", style={"opacity": "0.9"})
                ])
            ], className="kpi-card", style=DARK_CARD)
        ], width=12, lg=3, className="mb-4"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Best F1 Score", className="text-muted"),
                    html.H3(id="best-f1-score", style={"color": "#00d4ff", "font-weight": "700"}),
                    html.P(id="best-f1-model", className="text-muted mt-2 mb-0")
                ])
            ], style=DARK_CARD)
        ], width=12, lg=3, className="mb-4"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Best F2 Score", className="text-muted"),
                    html.H3(id="best-f2-score", style={"color": "#ff0055", "font-weight": "700"}),
                    html.P(id="best-f2-model", className="text-muted mt-2 mb-0")
                ])
            ], style=DARK_CARD)
        ], width=12, lg=3, className="mb-4"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Best AUC-PR", className="text-muted"),
                    html.H3(id="best-aucpr-score", style={"color": "#00d4ff", "font-weight": "700"}),
                    html.P(id="best-aucpr-model", className="text-muted mt-2 mb-0")
                ])
            ], style=DARK_CARD)
        ], width=12, lg=3, className="mb-4"),
    ], className="g-4"),
    
    # Performance Comparison Chart
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    [
                        html.Span("📊 ", style={"marginRight": "0.5rem"}),
                        "Model Performance Comparison"
                    ],
                    className="text-info",
                    style={"fontWeight": "600", "letterSpacing": "0.3px"}
                ),
                dbc.CardBody([
                    dcc.Graph(id="metrics-comparison", config={'displayModeBar': False})
                ])
            ], className="card", style=PANEL_STYLE if 'PANEL_STYLE' in globals() else {"background": "#141414", "border": "1px solid #2a2a2a", "borderRadius": "12px", "boxShadow": "0 4px 12px rgba(0,0,0,0.4)"})
        ], width=12, lg=8, className="mb-4"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    [
                        html.Span("🎯 ", style={"marginRight": "0.5rem"}),
                        "Confusion Matrix (Best Model)"
                    ],
                    className="text-danger",
                    style={"fontWeight": "600", "letterSpacing": "0.3px"}
                ),
                dbc.CardBody([
                    dcc.Graph(id="confusion-matrix", config={'displayModeBar': False})
                ])
            ], className="card", style=PANEL_STYLE if 'PANEL_STYLE' in globals() else {"background": "#141414", "border": "1px solid #2a2a2a", "borderRadius": "12px", "boxShadow": "0 4px 12px rgba(0,0,0,0.4)"})
        ], width=12, lg=4, className="mb-4"),
    ], className="g-4"),
    
    # ROC & PR Curves
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    [
                        html.Span("📈 ", style={"marginRight": "0.5rem"}),
                        "ROC Curve"
                    ],
                    className="text-info",
                    style={"fontWeight": "600", "letterSpacing": "0.3px"}
                ),
                dbc.CardBody([
                    dcc.Graph(id="roc-curve", config={'displayModeBar': False})
                ])
            ], className="card", style=PANEL_STYLE if 'PANEL_STYLE' in globals() else {"background": "#141414", "border": "1px solid #2a2a2a", "borderRadius": "12px", "boxShadow": "0 4px 12px rgba(0,0,0,0.4)"})
        ], width=12, lg=6, className="mb-4"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    [
                        html.Span("📉 ", style={"marginRight": "0.5rem"}),
                        "Precision-Recall Curve"
                    ],
                    className="text-danger",
                    style={"fontWeight": "600", "letterSpacing": "0.3px"}
                ),
                dbc.CardBody([
                    dcc.Graph(id="pr-curve", config={'displayModeBar': False})
                ])
            ], className="card", style=PANEL_STYLE if 'PANEL_STYLE' in globals() else {"background": "#141414", "border": "1px solid #2a2a2a", "borderRadius": "12px", "boxShadow": "0 4px 12px rgba(0,0,0,0.4)"})
        ], width=12, lg=6, className="mb-4"),
    ], className="g-4"),
    
    # Detailed Metrics Table
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    [
                        html.Span("📋 ", style={"marginRight": "0.5rem"}),
                        "Detailed Metrics"
                    ],
                    className="text-info",
                    style={"fontWeight": "600", "letterSpacing": "0.3px"}
                ),
                dbc.CardBody([
                    html.Div(id="metrics-table")
                ])
            ], className="card", style=PANEL_STYLE if 'PANEL_STYLE' in globals() else {"background": "#141414", "border": "1px solid #2a2a2a", "borderRadius": "12px", "boxShadow": "0 4px 12px rgba(0,0,0,0.4)"})
        ], width=12, className="mb-4"),
    ]),
])

@callback(
    [Output("best-model-name", "children"),
     Output("best-model-f2", "children"),
     Output("best-f1-score", "children"),
     Output("best-f1-model", "children"),
     Output("best-f2-score", "children"),
     Output("best-f2-model", "children"),
     Output("best-aucpr-score", "children"),
     Output("best-aucpr-model", "children"),
     Output("metrics-comparison", "figure"),
     Output("confusion-matrix", "figure"),
     Output("roc-curve", "figure"),
     Output("pr-curve", "figure"),
     Output("metrics-table", "children")],
    Input("metrics-comparison", "id")  # Dummy input to trigger on load
)
def update_model_metrics(_):
    metrics = get_model_metrics()
    
    # Find best model overall (by F2 score, which is the main metric)
    if metrics['f2_scores']:
        best_f2_idx = metrics['f2_scores'].index(max(metrics['f2_scores']))
        best_model_name = metrics['models'][best_f2_idx]
        best_f2_score = f"{max(metrics['f2_scores']):.3f}"
        best_model_f2_text = f"F2 Score: {best_f2_score}"
    else:
        best_model_name = "N/A"
        best_model_f2_text = "F2 Score: N/A"
    
    # Find best model for each metric
    if metrics['f1_scores']:
        best_f1_idx = metrics['f1_scores'].index(max(metrics['f1_scores']))
        best_f1_score = f"{max(metrics['f1_scores']):.3f}"
        best_f1_model = metrics['models'][best_f1_idx]
    else:
        best_f1_score = "N/A"
        best_f1_model = "N/A"
    
    if metrics['f2_scores']:
        best_f2_idx = metrics['f2_scores'].index(max(metrics['f2_scores']))
        best_f2_score = f"{max(metrics['f2_scores']):.3f}"
        best_f2_model = metrics['models'][best_f2_idx]
    else:
        best_f2_score = "N/A"
        best_f2_model = "N/A"
    
    if metrics['auc_pr']:
        best_aucpr_idx = metrics['auc_pr'].index(max(metrics['auc_pr']))
        best_aucpr_score = f"{max(metrics['auc_pr']):.3f}"
        best_aucpr_model = metrics['models'][best_aucpr_idx]
    else:
        best_aucpr_score = "N/A"
        best_aucpr_model = "N/A"
    
    # Metrics Comparison Bar Chart
    fig_metrics = go.Figure()
    fig_metrics.add_trace(go.Bar(
        x=metrics['models'],
        y=metrics['f1_scores'],
        name='F1 Score',
        marker_color='#00d4ff',
        marker_line_color='#00d4ff',
        marker_line_width=1.5,
        opacity=0.8
    ))
    fig_metrics.add_trace(go.Bar(
        x=metrics['models'],
        y=metrics['f2_scores'],
        name='F2 Score',
        marker_color='#ff0055',
        marker_line_color='#ff0055',
        marker_line_width=1.5,
        opacity=0.8
    ))
    
    fig_metrics.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter', 'color': 'white'},
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20),
        yaxis_title="Score"
    )
    
    # Confusion Matrix (XGBoost - best model)
    # Example values: [TN, FP], [FN, TP]
    cm_values = [[9500, 50], [10, 440]]
    fig_cm = px.imshow(
        cm_values,
        x=['Predicted Legit', 'Predicted Fraud'],
        y=['Actual Legit', 'Actual Fraud'],
        color_continuous_scale='Reds',
        text_auto=True,
        aspect='auto'
    )
    fig_cm.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter', 'color': 'white'},
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    # ROC Curve (mock data)
    fpr = np.linspace(0, 1, 100)
    tpr = 1 - np.exp(-5 * fpr)  # Mock ROC curve
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name='XGBoost (AUC=0.99)',
        line=dict(color='#00d4ff', width=3)
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='#666', width=2, dash='dash')
    ))
    fig_roc.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter', 'color': 'white'},
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # PR Curve (mock data)
    recall = np.linspace(0, 1, 100)
    precision = 0.9 + 0.1 * np.exp(-10 * (1 - recall))  # Mock PR curve
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode='lines',
        name='XGBoost (AUC-PR=0.97)',
        line=dict(color='#ff0055', width=3),
        fill='tozeroy',
        fillcolor='rgba(255, 0, 85, 0.2)'
    ))
    fig_pr.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter', 'color': 'white'},
        xaxis_title="Recall",
        yaxis_title="Precision",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # Metrics Table
    metrics_df = pd.DataFrame({
        'Model': metrics['models'],
        'F1 Score': [f"{x:.3f}" for x in metrics['f1_scores']],
        'F2 Score': [f"{x:.3f}" for x in metrics['f2_scores']],
        'Precision': [f"{x:.3f}" for x in metrics['precision']],
        'Recall': [f"{x:.3f}" for x in metrics['recall']],
        'AUC-PR': [f"{x:.3f}" for x in metrics['auc_pr']],
        'ROC-AUC': [f"{x:.3f}" for x in metrics['roc_auc']],
    })
    
    # Create table manually
    table_header = [html.Thead(html.Tr([html.Th(col) for col in metrics_df.columns]))]
    table_rows = [html.Tr([html.Td(metrics_df.iloc[i][col]) for col in metrics_df.columns]) 
                  for i in range(len(metrics_df))]
    table_body = [html.Tbody(table_rows)]
    table = dbc.Table(table_header + table_body, striped=True, bordered=True, 
                     hover=True, responsive=True, className="table-dark")
    
    return (best_model_name, best_model_f2_text,
            best_f1_score, best_f1_model, best_f2_score, best_f2_model, 
            best_aucpr_score, best_aucpr_model,
            fig_metrics, fig_cm, fig_roc, fig_pr, table)
