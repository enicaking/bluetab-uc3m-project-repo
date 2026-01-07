import dash
from dash import html, dcc, callback, Input, Output, State
from dash import callback_context
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/simulator', name="Fraud Simulator", order=3)

# Standard styling for our dark theme
DARK_CARD = {"background-color": "#111", "border": "1px solid #333", "padding": "20px", "border-radius": "10px"}

layout = dbc.Container(fluid=True, children=[
    html.H2("Real-Time Transaction Simulator", className="text-center my-4", style={"font-weight": "700"}),
    
    # We wrap the columns in a Row and center the content of that Row
    dbc.Row([
        
        # --- INPUT COLUMN (Width 5) ---
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
                    style={"color": "#000"} 
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
                        marks={0: '00:00', 6: '6:00', 12: '12:00', 18: '18:00', 23: '23:00'},
                        updatemode='drag'
                    ),
                ], className="mb-3"),

                dbc.Row([
                    dbc.Col(dbc.Button('Run Analysis', id='sim-button', color="primary", className="w-100", n_clicks=0), width=8),
                    dbc.Col(dbc.Button('Reset', id='reset-button', color="secondary", outline=True, className="w-100", n_clicks=0), width=4),
                ], className="mt-4")
            ])
        ], width=10), # Smaller width to fit side-by-side

        # --- OUTPUT COLUMN (Width 5) ---
        dbc.Col([
            html.Div(style={**DARK_CARD, "height": "100%", "display": "flex", "flex-direction": "column", "align-items": "center", "justify-content": "center", "text-align": "center"}, children=[
                html.Div(id='prediction-output')
            ])
        ], width=10)

    ], className="justify-content-center g-4") # This centers the two columns and adds a gap (g-4)
])

@callback(
    Output('prediction-output', 'children'),
    Output('amount-input', 'value'),
    Output('cust-country-input', 'value'),
    Output('merch-country-input', 'value'),
    Output('time-slider', 'value'),
    Input('sim-button', 'n_clicks'),
    Input('reset-button', 'n_clicks'), # New Input
    State('amount-input', 'value'),
    State('cust-country-input', 'value'),
    State('merch-country-input', 'value'),
    State('time-slider', 'value')
)
def predict_or_reset(sim_clicks, reset_clicks, amount, cust, merch, time_val):
    # Determine which button was actually clicked
    ctx = callback_context
    if not ctx.triggered:
        return html.H3("Awaiting Input...", style={"opacity": "0.5"}), 100, 'USA', 'USA', 12
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # --- RESET LOGIC ---
    if button_id == 'reset-button':
        return html.H3("Awaiting Input...", style={"opacity": "0.5"}), 100, 'USA', 'USA', 12

    # --- SIMULATION LOGIC ---
    if sim_clicks > 0:
        is_mismatch = cust != merch
        if amount > 5000 or (amount > 1000 and is_mismatch):
            result = html.Div([
                html.H1("⚠️ HIGH RISK", style={"color": "#ff0055", "text-shadow": "0 0 15px #ff0055"}),
                html.P(f"Transaction of ${amount} flagged.")
            ])
        else:
            result = html.Div([
                html.H1("✅ APPROVED", style={"color": "#00ffcc", "text-shadow": "0 0 15px #00ffcc"}),
                html.P("Transaction appears normal.")
            ])
        # Return the result + the current values (so they don't change when you click Analyze)
        return result, amount, cust, merch, time_val

    return html.H3("Awaiting Input...", style={"opacity": "0.5"}), 100, 'USA', 'USA', 12
