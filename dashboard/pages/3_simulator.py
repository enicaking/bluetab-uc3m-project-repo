"""
Real-Time Transaction Fraud Simulator
"""
import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils import load_data

dash.register_page(__name__, path='/simulator', name="Fraud Simulator", order=3)

DARK_STYLE = {
    "background-color": "#000000",
    "color": "#FFFFFF",
    "font-family": "'Inter', sans-serif",
}

DARK_CARD = {"background": "#111", "border": "1px solid #333", "border-radius": "10px", "padding": "20px"}

# Load model data for reference
df = load_data("df_exp_50_2.csv")

layout = dbc.Container(fluid=True, style=DARK_STYLE, children=[
    html.Link(href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700&display=swap", rel="stylesheet"),
    
    # Header
    dbc.Row([
        dbc.Col(html.H1("Real-Time Transaction Fraud Simulator", className="my-4", style={"font-weight": "700"}), width=12),
        dbc.Col(html.P("Test transaction scenarios and evaluate fraud risk in real-time", className="text-muted mb-4"), width=12)
    ]),
    
    dbc.Row([
        # Input Column
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Transaction Details", className="text-info"),
                dbc.CardBody([
                    html.Label("Transaction Amount ($)", className="text-muted mb-2"),
                    dbc.Input(
                        id='amount-input',
                        type='number',
                        value=100,
                        min=0,
                        step=0.01,
                        className="mb-3",
                        style={"background": "#000", "color": "white", "border": "1px solid #444"}
                    ),
                    
                    html.Label("Customer Country", className="text-muted mb-2"),
                    dcc.Dropdown(
                        id='cust-country-input',
                        options=[
                            {'label': 'USA', 'value': 'USA'},
                            {'label': 'Spain', 'value': 'Spain'},
                            {'label': 'UK', 'value': 'UK'},
                            {'label': 'France', 'value': 'France'},
                            {'label': 'Germany', 'value': 'Germany'},
                            {'label': 'China', 'value': 'China'},
                            {'label': 'Brazil', 'value': 'Brazil'},
                        ],
                        value='USA',
                        className="mb-3",
                        style={"color": "#000"}
                    ),
                    
                    html.Label("Merchant Country", className="text-muted mb-2"),
                    dcc.Dropdown(
                        id='merch-country-input',
                        options=[
                            {'label': 'USA', 'value': 'USA'},
                            {'label': 'Spain', 'value': 'Spain'},
                            {'label': 'UK', 'value': 'UK'},
                            {'label': 'France', 'value': 'France'},
                            {'label': 'Germany', 'value': 'Germany'},
                            {'label': 'China', 'value': 'China'},
                            {'label': 'Brazil', 'value': 'Brazil'},
                        ],
                        value='USA',
                        className="mb-3",
                        style={"color": "#000"}
                    ),
                    
                    html.Label("Time of Day", className="text-muted mb-2"),
                    html.Div([
                        html.Span(id='time-display', style={"color": "#00d4ff", "font-weight": "bold"}),
                        dcc.Slider(
                            0, 23, 1,
                            value=12,
                            id='time-slider',
                            marks={0: '00:00', 6: '6:00', 12: '12:00', 18: '18:00', 23: '23:00'},
                            updatemode='drag',
                            className="mt-2"
                        ),
                    ], className="mb-3"),
                    
                    html.Label("Device Type", className="text-muted mb-2"),
                    dcc.Dropdown(
                        id='device-input',
                        options=[
                            {'label': 'Mobile', 'value': 'mobile'},
                            {'label': 'Desktop', 'value': 'desktop'},
                            {'label': 'Tablet', 'value': 'tablet'},
                        ],
                        value='mobile',
                        className="mb-3",
                        style={"color": "#000"}
                    ),
                    
                    html.Label("Customer Age", className="text-muted mb-2"),
                    dbc.Input(
                        id='age-input',
                        type='number',
                        value=35,
                        min=18,
                        max=100,
                        className="mb-3",
                        style={"background": "#000", "color": "white", "border": "1px solid #444"}
                    ),
                    
                    dbc.Row([
                        dbc.Col(dbc.Button('Analyze Transaction', id='sim-button', color="primary", className="w-100", n_clicks=0), width=8),
                        dbc.Col(dbc.Button('Reset', id='reset-button', color="secondary", outline=True, className="w-100", n_clicks=0), width=4),
                    ], className="mt-4")
                ])
            ], style={"background": "#111", "border": "1px solid #333", "height": "100%"})
        ], width=12, lg=5, className="mb-4"),
        
        # Output Column
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Risk Assessment", className="text-danger"),
                dbc.CardBody([
                    html.Div(id='prediction-output', style={"min-height": "400px", "display": "flex", "align-items": "center", "justify-content": "center"})
                ])
            ], style={"background": "#111", "border": "1px solid #333", "height": "100%"})
        ], width=12, lg=7, className="mb-4"),
    ], className="g-4"),
    
    # Risk Factors Explanation
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Risk Factors", className="text-info"),
                dbc.CardBody([
                    html.Ul([
                        html.Li("High transaction amounts (>$5,000) increase risk"),
                        html.Li("Cross-border transactions (different countries) are riskier"),
                        html.Li("Transactions during unusual hours (night/early morning)"),
                        html.Li("New or unusual device patterns"),
                        html.Li("Customer age and credit score history")
                    ], style={"color": "#ccc"})
                ])
            ], style={"background": "#111", "border": "1px solid #333"})
        ], width=12, className="mb-4"),
    ]),
])

@callback(
    [Output('prediction-output', 'children'),
     Output('time-display', 'children'),
     Output('amount-input', 'value'),
     Output('cust-country-input', 'value'),
     Output('merch-country-input', 'value'),
     Output('time-slider', 'value'),
     Output('device-input', 'value'),
     Output('age-input', 'value')],
    [Input('sim-button', 'n_clicks'),
     Input('reset-button', 'n_clicks')],
    [State('amount-input', 'value'),
     State('cust-country-input', 'value'),
     State('merch-country-input', 'value'),
     State('time-slider', 'value'),
     State('device-input', 'value'),
     State('age-input', 'value')]
)
def predict_or_reset(sim_clicks, reset_clicks, amount, cust, merch, time_val, device, age):
    from dash import callback_context
    
    ctx = callback_context
    if not ctx.triggered:
        return (
            html.Div([
                html.H3("Awaiting Input...", style={"opacity": "0.5", "text-align": "center"})
            ]),
            f"{time_val:02d}:00",
            100, 'USA', 'USA', 12, 'mobile', 35
        )
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Reset
    if button_id == 'reset-button':
        return (
            html.Div([
                html.H3("Awaiting Input...", style={"opacity": "0.5", "text-align": "center"})
            ]),
            "12:00",
            100, 'USA', 'USA', 12, 'mobile', 35
        )
    
    # Simulation Logic
    if sim_clicks > 0:
        risk_score = 0
        risk_factors = []
        
        # Amount risk
        if amount > 5000:
            risk_score += 40
            risk_factors.append("High transaction amount")
        elif amount > 1000:
            risk_score += 20
            risk_factors.append("Moderate transaction amount")
        
        # Geographic risk
        is_foreign = cust != merch
        if is_foreign:
            risk_score += 30
            risk_factors.append("Cross-border transaction")
        
        # Time risk
        if time_val < 6 or time_val > 22:
            risk_score += 15
            risk_factors.append("Unusual time of day")
        
        # Device risk
        if device == 'mobile':
            risk_score += 5  # Mobile is slightly riskier
        
        # Age risk (younger customers)
        if age < 25:
            risk_score += 10
            risk_factors.append("Young customer profile")
        
        # Calculate probability (mock)
        fraud_prob = min(risk_score / 100, 0.95)
        
        # Determine result
        if fraud_prob > 0.7:
            result = html.Div([
                html.H1("⚠️ HIGH RISK", style={"color": "#ff0055", "text-align": "center", "text-shadow": "0 0 15px #ff0055", "margin-bottom": "20px"}),
                html.H3(f"Fraud Probability: {fraud_prob*100:.1f}%", style={"color": "#ff0055", "text-align": "center", "margin-bottom": "20px"}),
                html.P("Transaction flagged for manual review", style={"text-align": "center", "color": "#ccc"}),
                html.Hr(style={"border-color": "#333", "margin": "20px 0"}),
                html.H6("Risk Factors:", style={"color": "#00d4ff", "margin-top": "20px"}),
                html.Ul([html.Li(factor, style={"color": "#ccc", "margin": "5px 0"}) for factor in risk_factors])
            ])
        elif fraud_prob > 0.4:
            result = html.Div([
                html.H1("⚠️ MEDIUM RISK", style={"color": "#ffaa00", "text-align": "center", "text-shadow": "0 0 15px #ffaa00", "margin-bottom": "20px"}),
                html.H3(f"Fraud Probability: {fraud_prob*100:.1f}%", style={"color": "#ffaa00", "text-align": "center", "margin-bottom": "20px"}),
                html.P("Additional verification recommended", style={"text-align": "center", "color": "#ccc"}),
                html.Hr(style={"border-color": "#333", "margin": "20px 0"}),
                html.H6("Risk Factors:", style={"color": "#00d4ff", "margin-top": "20px"}),
                html.Ul([html.Li(factor, style={"color": "#ccc", "margin": "5px 0"}) for factor in risk_factors])
            ])
        else:
            result = html.Div([
                html.H1("✅ APPROVED", style={"color": "#00ffcc", "text-align": "center", "text-shadow": "0 0 15px #00ffcc", "margin-bottom": "20px"}),
                html.H3(f"Fraud Probability: {fraud_prob*100:.1f}%", style={"color": "#00ffcc", "text-align": "center", "margin-bottom": "20px"}),
                html.P("Transaction appears normal", style={"text-align": "center", "color": "#ccc"})
            ])
        
        return result, f"{time_val:02d}:00", amount, cust, merch, time_val, device, age
    
    return (
        html.Div([
            html.H3("Awaiting Input...", style={"opacity": "0.5", "text-align": "center"})
        ]),
        f"{time_val:02d}:00",
        100, 'USA', 'USA', 12, 'mobile', 35
    )
