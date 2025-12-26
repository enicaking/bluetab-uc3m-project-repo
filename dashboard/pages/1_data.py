import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd

# Using a Bootstrap theme as a baseline (CYBORG is dark)
dash.register_page(__name__, path='/', name="Dataset")

# --- Styling Constants ---
DARK_STYLE = {
    "background-color": "#000000",
    "color": "#FFFFFF",
    "font-family": "'Inter', sans-serif",
}

# --- Data Preparation (Same as before) ---
data = {
    "customer_country": ["USA", "USA", "Spain", "Spain", "Spain", "France", "UK", "UK", "Germany", "USA", "Germany"],
    "is_fraud": [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0]
}
df = pd.DataFrame(data)

# Pre-calculations
volume_df = df['customer_country'].value_counts().reset_index()
volume_df.columns = ['country', 'metric_value']
fraud_df = df.groupby('customer_country')['is_fraud'].mean().reset_index()
fraud_df.columns = ['country', 'metric_value']
fraud_df['metric_value'] *= 100

# --- Page Layout ---
layout = dbc.Container(fluid=True, style=DARK_STYLE, children=[
    # External Font Import
    html.Link(href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700&display=swap", rel="stylesheet"),
    
    dbc.Row([
        dbc.Col(html.H1("Dataset Breakdown", className="text-center my-4", 
                        style={"font-weight": "700", "letter-spacing": "-1px"}), width=12)
    ]),

    dbc.Row([
        dbc.Col([
            html.Div([
                html.Label("Metric Toggle", style={"margin-bottom": "10px", "opacity": "0.7"}),
                dcc.RadioItems(
                    id='map-metric-selector',
                    options=[
                        {'label': ' Total Volume', 'value': 'volume'},
                        {'label': ' Fraud Rate (%)', 'value': 'rate'}
                    ],
                    value='volume',
                    inline=True,
                    inputStyle={"margin-right": "10px", "margin-left": "20px"},
                    style={"padding": "10px", "border-radius": "8px", "background": "#111"}
                ),
            ], className="d-flex flex-column align-items-center")
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col(dcc.Graph(id='interactive-map', config={'displayModeBar': False}), width=12)
    ], style={"margin-top": "20px"})
])

# --- Callback ---
@callback(
    Output('interactive-map', 'figure'),
    Input('map-metric-selector', 'value')
)
def update_map(selected_metric):
    # (Logic for choosing df remains same)
    if selected_metric == 'volume':
        display_df, title, colorscale = volume_df, "Transaction Distribution", px.colors.sequential.Purp
    else:
        display_df, title, colorscale = fraud_df, "Regional Fraud Probability", ["#1a0000", "#ff0000"]

    fig = px.choropleth(
        display_df, locations="country", locationmode="country names",
        color="metric_value", hover_name="country", title=title
    )

    # Styling the figure to match the black background
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        geo=dict(
            bgcolor='rgba(0,0,0,0)',
            showlakes=True, lakecolor="#111",
            showcountries=True, countrycolor="#444"
        ),
        font={"family": "Inter", "color": "white"},
        title_x=0.5,
        margin={"r":0,"t":80,"l":0,"b":0}
    )
    fig.update_coloraxes(colorbar_len=0.5)
    
    return fig