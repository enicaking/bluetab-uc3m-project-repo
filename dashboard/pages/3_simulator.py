"""
Fraud Detection Simulator Page
Uses the trained RandomForest model to classify transactions in real-time
"""
import dash
from dash import html, dcc, callback, Input, Output, State
from dash import callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Design system
try:
    from design_system import (
        DARK_STYLE, PANEL_STYLE, KPI_CARD_STYLE,
        ACCENT_PRIMARY, ACCENT_DANGER, ACCENT_SUCCESS,
        get_gradient_text_style
    )
    ACCENT_TEAL = ACCENT_PRIMARY
    ACCENT_PINK = ACCENT_DANGER
    ACCENT_SUCCESS = ACCENT_SUCCESS
except ImportError:
    DARK_STYLE = {
        "background-color": "#0a0a0a",
        "color": "#FFFFFF",
        "font-family": "'Inter', sans-serif",
    }
    PANEL_STYLE = {"background": "#141414", "border": "1px solid #2a2a2a", "borderRadius": "12px", "padding": "24px"}
    ACCENT_TEAL = "#00d4ff"
    ACCENT_PINK = "#ff0055"
    ACCENT_SUCCESS = "#00ff88"

dash.register_page(__name__, path='/simulator', name="Fraud Simulator", order=3)

# Load model and get feature info
def load_model_and_features():
    """Load RandomForest model and return model + feature info"""
    models_dir = Path(__file__).parent.parent.parent / "content" / "models"
    model_path = models_dir / "rf_model_df_same_prop_new.pkl"
    
    if not model_path.exists():
        print(f"Warning: Model file not found at {model_path}")
        return None, None, None
    
    try:
        model = joblib.load(model_path)
        
        # Get expected features
        if hasattr(model, 'feature_names_in_'):
            expected_features = list(model.feature_names_in_)
        elif hasattr(model, 'feature_names_'):
            expected_features = list(model.feature_names_)
        else:
            print("Warning: Cannot determine model features")
            return model, None, None
        
        # Load dataset to get statistics for neutral values
        data_dir = Path(__file__).parent.parent.parent / "content"
        csv_file = data_dir / "df_same_prop_new.csv"
        
        df_stats = None
        

        if csv_file.exists():

            chunk_size = 50000
            chunks = []
            try:
                for chunk in pd.read_csv(
                    csv_file,
                    chunksize=chunk_size,
                    nrows=chunk_size * 2,
                    engine="python",
                    on_bad_lines="warn"  # or "skip"
                ):
                    chunks.append(chunk)
                    if len(chunks) >= 2:
                        break
                if chunks:
                    df_stats = pd.concat(chunks, ignore_index=True)
                else:
                    df_stats = pd.read_csv(
                        csv_file,
                        nrows=50000,
                        engine="python",
                        on_bad_lines="warn"
                    )
            except Exception:
                df_stats = pd.read_csv(
                    csv_file,
                    nrows=50000,
                    engine="python",
                    on_bad_lines="warn"
                )
            print(f"Loaded {len(df_stats)} rows")
        else:
            print(f"Warning: Dataset file not found at {csv_file}")
        
        print(f"Model loaded successfully. Expected {len(expected_features)} features")
        return model, expected_features, df_stats
    except Exception as e:
        import traceback
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return None, None, None

# Load model and dataset
MODEL, EXPECTED_FEATURES, DF_STATS = load_model_and_features()
country_features = [f for f in EXPECTED_FEATURES if f.startswith('customer_country_')]
cleaned_country_features = [feature.replace('customer_country_', '') for feature in country_features]
formatted_countries = [{'label': country, 'value': country} for country in cleaned_country_features]

def prepare_input_data(amount, customer_country, expected_features, df_stats=None):
    """
    Prepare input data for the RandomForest model
    Based on eval_desbalanceado_rf_all_vars from modelos.ipynb
    """
    # Create base dataframe with available features after one-hot encoding
    base_data = {}

    # Add V variables (PCA features) 
    v_features = [f for f in expected_features if f.startswith("V") and f[1:].isdigit()]
    # Default values are neutral 
    for v in v_features:
        base_data[v] = 0.0

    # Inject fraud-leaning PCA values but keep it standard
    if MODEL is not None:
        fraud_pca = {'V14': -1.400272740503877, 'V17': 1.26972022761454, 'V3': -2.1380155662026246, 'V21': 0.5331025507790222, 'V27': 0.38669295421558797}
        for v, val in fraud_pca.items():
            base_data[v] = val

    
    # Add amount_log if needed
    if 'amount_log' in expected_features:
        base_data['amount_log'] = float(np.log1p(amount))
    elif 'amount' in expected_features:
        base_data['amount'] = float(amount)
    
    # Handle customer_country (one-hot encoded)
    # The model expects one-hot encoded country features like 'customer_country_Algeria', etc.
    country_features = [f for f in expected_features if f.startswith('customer_country_')]
    cleaned_country_features = [feature.replace('customer_country_', '') for feature in country_features]

    # Set all country features to 0, then set the selected one to 1
    for country_feat in country_features:
        base_data[country_feat] = 0
    
    # Find matching country feature (case-insensitive partial match)
    customer_country_clean = str(customer_country).strip()
    country_matched = False
    
    if country_features:
        # Normalize country name (handle variations)
        country_normalized = customer_country_clean.lower().replace('united states of america', 'usa').replace('united kingdom', 'uk')
        
        for country_feat in country_features:
            # Extract country name from feature name (e.g., 'customer_country_Algeria' -> 'Algeria')
            country_name = country_feat.replace('customer_country_', '').lower()
            if country_name == country_normalized or \
               country_name in country_normalized or \
               country_normalized in country_name or \
               (country_normalized == 'usa' and 'usa' in country_name) or \
               (country_normalized == 'uk' and ('united_kingdom' in country_name or 'uk' in country_name)):
                base_data[country_feat] = 1
                country_matched = True
                break
        
        # If no exact match found, try fuzzy matching with underscores/spaces
        if not country_matched:
            for country_feat in country_features:
                country_name = country_feat.replace('customer_country_', '').lower().replace('_', ' ')
                if country_name == customer_country_clean.lower() or \
                   country_name in customer_country_clean.lower() or \
                   customer_country_clean.lower() in country_name:
                    base_data[country_feat] = 1
                    country_matched = True
                    break
        
        # If still no match, default to first country (or could use most common)
        if not country_matched and country_features:
            # Use the first country feature as default (often USA or most common)
            base_data[country_features[0]] = 1

    
    # Create DataFrame with exact feature order expected by model
    input_df = pd.DataFrame([base_data])
    
    # Ensure all expected features are present and in correct order
    missing_features = [f for f in expected_features if f not in input_df.columns]
    if missing_features:
        # Add missing features with default values
        for feat in missing_features:
            input_df[feat] = 0.0
    
    # Reorder columns to match expected features exactly
    input_df = input_df[expected_features]
    
    return input_df

layout = dbc.Container(fluid=True, style=DARK_STYLE, children=[
    html.Link(href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700&display=swap", rel="stylesheet"),
    
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Fraud Detection Simulator", className="my-4", style={"fontWeight": "700", "fontSize": "2.5rem"}),
            html.P("Real-time transaction classification using our trained RandomForest model", 
                  className="text-muted mb-4"),
            html.Div(
                dbc.Alert(
                    [
                        html.I(className="bi bi-info-circle me-2"),
                        "This simulator uses the RandomForest model trained on df_same_prop dataset. "
                        "Enter transaction details below to get a real-time fraud prediction."
                    ],
                    color="info",
                    className="mb-4"
                ),
                id="info-alert"
            ) if MODEL is None else html.Div()
        ], width=12)
    ]),
    
    # Main content
    dbc.Row([
        # Input Column
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Span("📝 ", style={"marginRight": "0.5rem"}),
                    "Transaction Details"
                ], className="text-info", style={"fontWeight": "600", "fontSize": "1.1rem"}),
                dbc.CardBody([
                    html.Label("Transaction Amount ($)", className="text-muted mb-2", style={"fontWeight": "600"}),
                    dbc.Input(
                        id='amount-input', 
                        type='number', 
                        value=100, 
                        min=0,
                        step=0.01,
                        className="mb-3", 
                        style={
                            "background": "#1a1a1a", 
                            "color": "white",
                            "border": "1px solid #333",
                            "borderRadius": "8px",
                            "padding": "12px"
                        }
                    ),
                    
                    html.Label("Customer Country", className="text-muted mb-2", style={"fontWeight": "600"}),
                    dcc.Dropdown(
                        id='cust-country-input',
                        options=formatted_countries,
                        value='United States of America',
                        className="mb-3",
                        style={
                            "backgroundColor": "#1a1a1a",
                            "color": "#ffffff",
                            "border": "1px solid #333",
                            "borderRadius": "8px"
                        },
                    ),
                    html.P("Note: The model uses one-hot encoded country features. Type to look up.", 
                          className="text-muted mb-3", style={"fontSize": "0.8rem", "fontStyle": "italic"}),
                    
                    html.Label("Time of Day (Hour)", className="text-muted mb-2", style={"fontWeight": "600"}),
                    html.Div([
                        dcc.Slider(
                            0, 23, 1, 
                            value=12, 
                            id='time-slider', 
                            marks={i: str(i) for i in [0, 6, 12, 18, 23]},
                            updatemode='drag',
                            tooltip={"placement": "bottom", "always_visible": True},
                            className="mb-3"
                        ),
                        html.Div(id='time-display', className="text-center text-muted mb-3", 
                                style={"fontSize": "0.9rem"})
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                '🔍 Analyze Transaction', 
                                id='sim-button', 
                                color="primary", 
                                className="w-100", 
                                n_clicks=0,
                                style={
                                    "background": ACCENT_TEAL,
                                    "border": "none",
                                    "borderRadius": "8px",
                                    "padding": "12px",
                                    "fontWeight": "600",
                                    "fontSize": "1rem"
                                }
                            )
                        ], width=8),
                        dbc.Col([
                            dbc.Button(
                                '🔄 Reset', 
                                id='reset-button', 
                                color="secondary", 
                                outline=True, 
                                className="w-100", 
                                n_clicks=0,
                                style={
                                    "borderRadius": "8px",
                                    "padding": "12px",
                                    "fontWeight": "600"
                                }
                            )
                        ], width=4),
                    ], className="mt-4 g-2"),
                ])
            ], style=PANEL_STYLE)
        ], width=12, lg=6, className="mb-4"),
        
        # Output Column
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Span("🎯 ", style={"marginRight": "0.5rem"}),
                    "Prediction Result"
                ], className="text-danger", style={"fontWeight": "600", "fontSize": "1.1rem"}),
                dbc.CardBody([
                    html.Div(
                        id='prediction-output',
                        style={
                            "minHeight": "400px",
                            "display": "flex",
                            "flexDirection": "column",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "textAlign": "center",
                            "padding": "40px"
                        }
                    )
                ])
            ], style=PANEL_STYLE)
        ], width=12, lg=6, className="mb-4"),
    ], className="g-4"),
    
    # Model Info Card
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Model Information", className="text-muted mb-3", style={"fontWeight": "600"}),
                    html.Div(id="model-info-display")
                ])
            ], style=PANEL_STYLE)
        ], width=12, className="mb-4")
    ])
])

@callback(
    [Output('prediction-output', 'children'),
     Output('amount-input', 'value'),
     Output('cust-country-input', 'value'),
     Output('time-slider', 'value'),
     Output('time-display', 'children'),
     Output('model-info-display', 'children')],
    [Input('sim-button', 'n_clicks'),
     Input('reset-button', 'n_clicks'),
     Input('time-slider', 'value')],
    [State('amount-input', 'value'),
     State('cust-country-input', 'value'),
     State('time-slider', 'value')]
)
def update_simulator(sim_clicks, reset_clicks, time_val_slider, amount, cust_country, time_val_state):
    """Handle simulation and reset actions"""
    ctx = callback_context
    if not ctx.triggered:
        time_str = f"{time_val_state:02d}:00"
        model_info = html.Div([
            html.P(f"Model: RandomForest (rf_model_df_same_prop_new.pkl)", className="mb-1 text-muted"),
            html.P(f"Features: {len(EXPECTED_FEATURES) if EXPECTED_FEATURES else 'N/A'} expected features", className="mb-1 text-muted"),
            html.P("Status: Ready", className="mb-0 text-success") if MODEL else html.P("Status: Model not loaded", className="mb-0 text-danger")
        ])
        return (
            html.Div([
                html.H3("Awaiting Input...", style={"opacity": "0.5", "color": "#888"}),
                html.P("Enter transaction details and click 'Analyze Transaction'", className="text-muted mt-3")
            ]),
            100, 'United States of America', 12, time_str, model_info
        )
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    time_val = time_val_slider if button_id == 'time-slider' else time_val_state
    
    # Update time display
    time_str = f"{time_val:02d}:00"
    
    # Model info
    model_info = html.Div([
        html.P(f"Model: RandomForest (rf_model_df_same_prop_new.pkl)", className="mb-1 text-muted", style={"fontSize": "0.9rem"}),
        html.P(f"Features: {len(EXPECTED_FEATURES) if EXPECTED_FEATURES else 'N/A'} expected features", className="mb-1 text-muted", style={"fontSize": "0.9rem"}),
        html.P("Status: Ready", className="mb-0 text-success", style={"fontSize": "0.9rem"}) if MODEL else html.P("Status: Model not loaded", className="mb-0 text-danger", style={"fontSize": "0.9rem"})
    ])
    
    # Reset logic
    if button_id == 'reset-button':
        return (
            html.Div([
                html.H3("Awaiting Input...", style={"opacity": "0.5", "color": "#888"}),
                html.P("Enter transaction details and click 'Analyze Transaction'", className="text-muted mt-3")
            ]),
            100, 'United States of America', 12, time_str, model_info
        )
    
    # Update time display when slider changes
    if button_id == 'time-slider':
        return (
            html.Div([
                html.H3("Awaiting Input...", style={"opacity": "0.5", "color": "#888"}),
                html.P("Enter transaction details and click 'Analyze Transaction'", className="text-muted mt-3")
            ]),
            amount if amount is not None else 100, 
            cust_country if cust_country is not None else 'United States of America', 
            time_val if time_val is not None else 12, 
            time_str, model_info
        )
    
    # Simulation logic
    if button_id == 'sim-button' and sim_clicks > 0:
        if MODEL is None or EXPECTED_FEATURES is None:
            return (
                html.Div([
                    html.H2("❌ Error", style={"color": ACCENT_PINK}),
                    html.P("Model could not be loaded. Please check that rf_model_df_same_prop_new.pkl exists in content/models/", 
                          className="text-muted mt-3")
                ]),
                amount, cust_country, time_val, time_str, model_info
            )
        
        try:
            # Validate inputs
            if amount is None or amount < 0:
                return (
                    html.Div([
                        html.H2("⚠️ Invalid Input", style={"color": ACCENT_PINK}),
                        html.P("Please enter a valid transaction amount", className="text-muted mt-3")
                    ]),
                    amount, cust_country, time_val, time_str, model_info
                )
            
            # Prepare input data
            input_df = prepare_input_data(amount, cust_country, EXPECTED_FEATURES, DF_STATS)
            
            # Make prediction
            prediction = MODEL.predict(input_df)[0]
            probabilities = MODEL.predict_proba(input_df)[0]
            fraud_probability = float(probabilities[1]) * 100  # Probability of fraud (class 1)
            safe_probability = float(probabilities[0]) * 100   # Probability of safe (class 0)
            
            # Determine result
            is_fraud = prediction == 1
            is_amber = (not is_fraud) and (fraud_probability >= 24)
            
            # Create result display
            if is_fraud:
                result = html.Div([
                    html.Div([
                        html.I(className="bi bi-exclamation-triangle-fill", style={"fontSize": "4rem", "color": ACCENT_PINK}),
                    ], className="mb-4"),
                    html.H1("⚠️ FRAUD DETECTED", 
                           style={
                               "color": ACCENT_PINK, 
                               "fontWeight": "700",
                               "fontSize": "2.5rem",
                               "textShadow": f"0 0 20px {ACCENT_PINK}",
                               "marginBottom": "20px"
                           }),
                    html.P(f"Transaction Amount: ${amount:,.2f}", className="text-muted mb-2", style={"fontSize": "1.1rem"}),
                    html.P(f"Customer Country: {cust_country}", className="text-muted mb-4", style={"fontSize": "1.1rem"}),
                    html.Hr(style={"borderColor": "#333", "margin": "20px 0"}),
                    html.H4("Confidence Metrics", className="text-muted mb-3", style={"fontWeight": "600"}),
                    html.Div([
                        dbc.Progress(
                            value=fraud_probability,
                            label=f"Fraud Probability: {fraud_probability:.2f}%",
                            color="danger",
                            className="mb-3",
                            style={"height": "30px", "fontSize": "0.9rem"}
                        ),
                        dbc.Progress(
                            value=safe_probability,
                            label=f"Safe Probability: {safe_probability:.2f}%",
                            color="success",
                            className="mb-3",
                            style={"height": "30px", "fontSize": "0.9rem"}
                        ),
                    ]),
                    html.P("Recommendation: Block this transaction and flag for manual review", 
                          className="text-danger mt-4", style={"fontWeight": "600", "fontSize": "1rem"})
                ])

            elif is_amber:
                # 🟧 AMBER — ALERT
                result = html.Div([
                    html.Div([
                        html.I(className="bi bi-exclamation-circle-fill",
                            style={"fontSize": "4rem", "color": "#f0ad4e"}),
                    ], className="mb-4"),

                    html.H1("⚠️ TRANSACTION ALERT",
                            style={
                                "color": "#f0ad4e",
                                "fontWeight": "700",
                                "fontSize": "2.5rem",
                                "textShadow": "0 0 15px rgba(240,173,78,0.6)",
                                "marginBottom": "20px"
                            }),

                    html.P(f"Transaction Amount: ${amount:,.2f}", className="text-muted mb-2"),
                    html.P(f"Customer Country: {cust_country}", className="text-muted mb-4"),

                    html.Hr(),

                    html.H4("Confidence Metrics", className="text-muted mb-3"),

                    dbc.Progress(
                        value=fraud_probability,
                        label=f"Fraud Probability: {fraud_probability:.2f}%",
                        color="warning",
                        className="mb-3",
                        style={"height": "30px"}
                    ),

                    dbc.Progress(
                        value=safe_probability,
                        label=f"Safe Probability: {safe_probability:.2f}%",
                        color="success",
                        className="mb-3",
                        style={"height": "30px"}
                    ),

                    html.P(
                        "Recommendation: Allow transaction but flag for monitoring",
                        className="text-warning mt-4",
                        style={"fontWeight": "600"}
                    )
                ])

            else:
                result = html.Div([
                    html.Div([
                        html.I(className="bi bi-check-circle-fill", style={"fontSize": "4rem", "color": ACCENT_SUCCESS}),
                    ], className="mb-4"),
                    html.H1("✅ TRANSACTION APPROVED", 
                           style={
                               "color": ACCENT_SUCCESS, 
                               "fontWeight": "700",
                               "fontSize": "2.5rem",
                               "textShadow": f"0 0 20px {ACCENT_SUCCESS}",
                               "marginBottom": "20px"
                           }),
                    html.P(f"Transaction Amount: ${amount:,.2f}", className="text-muted mb-2", style={"fontSize": "1.1rem"}),
                    html.P(f"Customer Country: {cust_country}", className="text-muted mb-4", style={"fontSize": "1.1rem"}),
                    html.Hr(style={"borderColor": "#333", "margin": "20px 0"}),
                    html.H4("Confidence Metrics", className="text-muted mb-3", style={"fontWeight": "600"}),
                    html.Div([
                        dbc.Progress(
                            value=fraud_probability,
                            label=f"Fraud Probability: {fraud_probability:.2f}%",
                            color="warning",
                            className="mb-3",
                            style={"height": "30px", "fontSize": "0.9rem"}
                        ),
                        dbc.Progress(
                            value=safe_probability,
                            label=f"Safe Probability: {safe_probability:.2f}%",
                            color="success",
                            className="mb-3",
                            style={"height": "30px", "fontSize": "0.9rem"}
                        ),
                    ]),
                    html.P("Recommendation: Proceed with transaction", 
                          className="text-success mt-4", style={"fontWeight": "600", "fontSize": "1rem"})
                ])
            
            return result, amount, cust_country, time_val, time_str, model_info
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            traceback.print_exc()
            return (
                html.Div([
                    html.H2("❌ Prediction Error", style={"color": ACCENT_PINK}),
                    html.P(f"An error occurred: {error_msg}", className="text-muted mt-3"),
                    html.P("Please check the model and input data format", className="text-muted")
                ]),
                amount, cust_country, time_val, time_str, model_info
            )
    
    # Default return
    return (
        html.Div([
            html.H3("Awaiting Input...", style={"opacity": "0.5", "color": "#888"}),
            html.P("Enter transaction details and click 'Analyze Transaction'", className="text-muted mt-3")
        ]),
        amount, cust_country, time_val, time_str, model_info
    )
