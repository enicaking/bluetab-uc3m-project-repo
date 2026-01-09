"""
Strategies & Recommendations Page
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
from utils import load_data, calculate_roi_metrics_auto

dash.register_page(__name__, path='/strategies', name="Strategies & ROI", order=4)

DARK_STYLE = {
    "background-color": "#000000",
    "color": "#FFFFFF",
    "font-family": "'Inter', sans-serif",
}

DARK_CARD = {"background": "#111", "border": "1px solid #333", "border-radius": "10px", "padding": "20px"}

# Load data for ROI calculations
df = load_data("df_exp_50.csv")
roi_metrics = calculate_roi_metrics_auto(df, model_name="LightGBM") if df is not None else {}

layout = dbc.Container(fluid=True, style=DARK_STYLE, children=[
    html.Link(href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700&display=swap", rel="stylesheet"),
    
    # Header
    dbc.Row([
        dbc.Col(html.H1("Strategies & Recommendations", className="my-4", style={"font-weight": "700"}), width=12),
        dbc.Col(html.P("Actionable insights and ROI analysis for fraud prevention", className="text-muted mb-4"), width=12)
    ]),
    
    # ROI Metrics Cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Total Fraud Amount", className="text-muted"),
                    html.H2(f"${roi_metrics.get('total_fraud_amount', 0):,.0f}", 
                           style={"color": "#ff0055", "font-weight": "700"}),
                    html.P("Potential loss without detection", className="text-muted mt-2 mb-0", style={"font-size": "0.85rem"})
                ])
            ], style=DARK_CARD)
        ], width=12, lg=3, className="mb-4"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Detection Improvement", className="text-muted"),
                    html.H2(f"+{roi_metrics.get('detection_rate_improvement', 0):.1f}%", 
                           style={"color": "#00d4ff", "font-weight": "700"}),
                    html.P("With ML model implementation", className="text-muted mt-2 mb-0", style={"font-size": "0.85rem"})
                ])
            ], style=DARK_CARD)
        ], width=12, lg=3, className="mb-4"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Potential Savings", className="text-muted"),
                    html.H2(f"${roi_metrics.get('potential_savings', 0):,.0f}", 
                           style={"color": "#00ffcc", "font-weight": "700"}),
                    html.P("Annual estimated savings", className="text-muted mt-2 mb-0", style={"font-size": "0.85rem"})
                ])
            ], style=DARK_CARD)
        ], width=12, lg=3, className="mb-4"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Investigation Cost", className="text-muted"),
                    html.H2(f"${roi_metrics.get('total_investigation_cost', 0):,.0f}", 
                           style={"color": "#ffaa00", "font-weight": "700"}),
                    html.P("Total operational cost", className="text-muted mt-2 mb-0", style={"font-size": "0.85rem"})
                ])
            ], style=DARK_CARD)
        ], width=12, lg=3, className="mb-4"),
    ], className="g-4"),
    
    # ROI Visualization
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("ROI Analysis", className="text-info"),
                dbc.CardBody([
                    dcc.Graph(id="roi-chart", config={'displayModeBar': False})
                ])
            ], style={"background": "#111", "border": "1px solid #333"})
        ], width=12, lg=8, className="mb-4"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Cost-Benefit Breakdown", className="text-danger"),
                dbc.CardBody([
                    html.Div(id="cost-benefit")
                ])
            ], style={"background": "#111", "border": "1px solid #333"})
        ], width=12, lg=4, className="mb-4"),
    ], className="g-4"),
    
    # Strategies Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Prevention Strategies", className="text-info"),
                dbc.CardBody([
                    html.H5("1. Real-Time Monitoring", className="text-info mb-3"),
                    html.P("Implement continuous monitoring of transactions using the ML model. Flag high-risk transactions for immediate review.", 
                          className="text-muted mb-4"),
                    
                    html.H5("2. Risk-Based Authentication", className="text-info mb-3"),
                    html.P("Apply multi-factor authentication for transactions exceeding risk thresholds. Use device fingerprinting and behavioral analysis.", 
                          className="text-muted mb-4"),
                    
                    html.H5("3. Geographic Segmentation", className="text-info mb-3"),
                    html.P("Monitor cross-border transactions more closely. Implement country-specific risk rules based on historical fraud patterns.", 
                          className="text-muted mb-4"),
                    
                    html.H5("4. Temporal Pattern Analysis", className="text-info mb-3"),
                    html.P("Identify and flag transactions occurring during unusual hours or deviating from customer's typical behavior patterns.", 
                          className="text-muted mb-4"),
                    
                    html.H5("5. Customer Segmentation", className="text-info mb-3"),
                    html.P("Segment customers by risk profile. Apply stricter controls for new customers or those with limited transaction history.", 
                          className="text-muted mb-0"),
                ])
            ], style={"background": "#111", "border": "1px solid #333"})
        ], width=12, lg=6, className="mb-4"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Operational Recommendations", className="text-danger"),
                dbc.CardBody([
                    html.H5("1. Automated Review Workflow", className="text-danger mb-3"),
                    html.P("Create tiered review process: High-risk (auto-block), Medium-risk (manual review), Low-risk (auto-approve).", 
                          className="text-muted mb-4"),
                    
                    html.H5("2. Model Retraining Schedule", className="text-danger mb-3"),
                    html.P("Retrain models monthly with new transaction data. Monitor model drift and performance degradation over time.", 
                          className="text-muted mb-4"),
                    
                    html.H5("3. Feedback Loop Integration", className="text-danger mb-3"),
                    html.P("Incorporate investigation outcomes back into the model. Use confirmed fraud cases to improve future predictions.", 
                          className="text-muted mb-4"),
                    
                    html.H5("4. Resource Allocation", className="text-danger mb-3"),
                    html.P("Optimize investigation team allocation based on model predictions. Focus resources on highest-risk cases.", 
                          className="text-muted mb-4"),
                    
                    html.H5("5. Compliance & Reporting", className="text-danger mb-3"),
                    html.P("Maintain detailed logs of all flagged transactions. Generate regular reports for regulatory compliance and audit purposes.", 
                          className="text-muted mb-0"),
                ])
            ], style={"background": "#111", "border": "1px solid #333"})
        ], width=12, lg=6, className="mb-4"),
    ], className="g-4"),
    
    # Implementation Roadmap
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Implementation Roadmap", className="text-info"),
                dbc.CardBody([
                    html.Div([
                        html.H6("Phase 1: Foundation (Weeks 1-4)", className="text-info mb-2"),
                        html.Ul([
                            html.Li("Deploy ML model to production environment"),
                            html.Li("Integrate with transaction processing system"),
                            html.Li("Set up monitoring and alerting infrastructure"),
                        ], className="text-muted mb-4"),
                        
                        html.H6("Phase 2: Optimization (Weeks 5-8)", className="text-info mb-2"),
                        html.Ul([
                            html.Li("Fine-tune risk thresholds based on business requirements"),
                            html.Li("Implement automated review workflows"),
                            html.Li("Train operations team on new system"),
                        ], className="text-muted mb-4"),
                        
                        html.H6("Phase 3: Enhancement (Weeks 9-12)", className="text-info mb-2"),
                        html.Ul([
                            html.Li("Add advanced features (behavioral analysis, device fingerprinting)"),
                            html.Li("Establish feedback loop for continuous improvement"),
                            html.Li("Generate comprehensive reporting and analytics"),
                        ], className="text-muted mb-0"),
                    ])
                ])
            ], style={"background": "#111", "border": "1px solid #333"})
        ], width=12, className="mb-4"),
    ]),
])

@callback(
    [Output("roi-chart", "figure"),
     Output("cost-benefit", "children")],
    Input("roi-chart", "id")
)
def update_roi_visualization(_):
    from utils import calculate_roi_metrics_auto, load_data
    roi_metrics = calculate_roi_metrics_auto(load_data("df_exp_50.csv"), model_name="LightGBM")
    
    # ROI Chart
    categories = ['Fraud Amount', 'Potential Savings', 'Investigation Cost', 'Net Benefit']
    values = [
        roi_metrics.get('total_fraud_amount', 0),
        roi_metrics.get('potential_savings', 0),
        roi_metrics.get('total_investigation_cost', 0),
        roi_metrics.get('potential_savings', 0) - roi_metrics.get('total_investigation_cost', 0)
    ]
    colors = ['#ff0055', '#00ffcc', '#ffaa00', '#00d4ff']
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f"${v:,.0f}" for v in values],
        textposition='outside'
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter', 'color': 'white'},
        yaxis_title="Amount ($)",
        margin=dict(l=20, r=20, t=40, b=40),
        showlegend=False
    )
    
    # Cost-Benefit Breakdown
    net_benefit = roi_metrics.get('potential_savings', 0) - roi_metrics.get('total_investigation_cost', 0)
    roi_percentage = (net_benefit / roi_metrics.get('total_investigation_cost', 1)) * 100 if roi_metrics.get('total_investigation_cost', 0) > 0 else 0
    
    cost_benefit = html.Div([
        html.H6("Net Benefit", className="text-success mb-3"),
        html.H3(f"${net_benefit:,.0f}", style={"color": "#00ffcc", "font-weight": "700"}),
        html.Hr(style={"border-color": "#333", "margin": "20px 0"}),
        html.H6("ROI", className="text-info mb-2"),
        html.H4(f"{roi_percentage:.1f}%", style={"color": "#00d4ff"}),
        html.Hr(style={"border-color": "#333", "margin": "20px 0"}),
        html.P("Breakdown:", className="text-muted mb-2"),
        html.Ul([
            html.Li(f"Savings: ${roi_metrics.get('potential_savings', 0):,.0f}", style={"color": "#ccc", "margin": "5px 0"}),
            html.Li(f"Costs: ${roi_metrics.get('total_investigation_cost', 0):,.0f}", style={"color": "#ccc", "margin": "5px 0"}),
        ], style={"font-size": "0.9rem"})
    ])
    
    return fig, cost_benefit

