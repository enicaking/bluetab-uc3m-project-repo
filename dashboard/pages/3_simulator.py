import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc

dash.register_page(__name__, name="Simulator")

# Standard styling for our dark theme
DARK_CARD = {"background-color": "#111", "border": "1px solid #333", "padding": "20px", "border-radius": "10px"}

layout = dbc.Container(fluid=True, children=[
    html.H2("Real-Time Transaction Simulator", className="my-4", style={"font-weight": "700"}),
    
    dbc.Row([
        # --- INPUT COLUMN ---
        dbc.Col([
            html.Div(style=DARK_CARD, children=[
                html.Label("Transaction Amount ($)", className="text-muted"),
                dbc.Input(id='amount-input', type='number', value=100, className="mb-3", style={"background": "#000", "color": "white"}),

                html.Label("Customer Country", className="text-muted"),
                dcc.Dropdown(
                    id='cust-country-input',
                    options=['USA', 'Spain', 'UK', 'France', 'Germany', 'China', 'Brazil'],
                    value='USA',
                    className="mb-3",
                    style={"color": "#000"} # Dropdown text usually needs to be dark for readability
                ),

                html.Label("Merchant Country", className="text-muted"),
                dcc.Dropdown(
                    id='merch-country-input',
                    options=['USA', 'Spain', 'UK', 'France', 'Germany', 'China', 'Brazil'],
                    value='USA',
                    className="mb-3",
                    style={"color": "#000"}
                ),

                html.Div([
                html.Label("Time of Day: ", className="text-muted"),
                html.Span(id='time-display', style={"color": "#00d4ff", "font-weight": "bold", "margin-left": "10px"}),
                dcc.Slider(
                    0, 23, 1, 
                    value=12, 
                    id='time-slider', 
                    marks={0: '00:00', 12: '12:00', 23: '23:00'},
                    updatemode='drag' # This makes the value update as you slide, not just when you let go
                ),
            ], className="mb-3"),

            dbc.Button('Run Fraud Analysis', id='sim-button', color="primary", className="mt-4 w-100", n_clicks=0),
            ])
        ], width=4),

        # --- OUTPUT COLUMN ---
        dbc.Col([
            html.Div(style={**DARK_CARD, "height": "100%", "display": "flex", "align-items": "center", "justify-content": "center", "text-align": "center"}, children=[
                html.Div(id='prediction-output')
            ])
        ], width=8)
    ])
])

@callback(
    Output('prediction-output', 'children'),
    Input('sim-button', 'n_clicks'),
    State('amount-input', 'value'),
    State('cust-country-input', 'value'),
    State('merch-country-input', 'value'),
    State('time-slider', 'value')
)
def predict_fraud(n_clicks, amount, cust_country, merch_country, time_val):
    if n_clicks > 0:
        # LOGIC: High risk if countries don't match or amount is very high
        is_mismatch = cust_country != merch_country
        
        if amount > 5000 or (amount > 1000 and is_mismatch):
            return html.Div([
                html.H1("⚠️ HIGH RISK", style={"color": "#ff0055", "text-shadow": "0 0 15px #ff0055"}),
                html.P(f"Analysis: Transaction of ${amount} from {cust_country} to {merch_country} at {time_val}:00 flagged for review.")
            ])
        else:
            return html.Div([
                html.H1("✅ APPROVED", style={"color": "#00ffcc", "text-shadow": "0 0 15px #00ffcc"}),
                html.P("Transaction pattern appears normal.")
            ])
            
    return html.H3("Awaiting Input...", style={"opacity": "0.5"})

@callback(
    Output('time-display', 'children'),
    Input('time-slider', 'value')
)
def update_time_display(selected_hour):
    # Formats the number into a 00:00 string
    return f"{selected_hour:02d}:00"