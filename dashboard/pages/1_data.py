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

# LAYOUT
# --- Page Layout ---
layout = dbc.Container(fluid=True, style=DARK_STYLE, children=[
    html.Link(href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700&display=swap", rel="stylesheet"),
    
    # Title Row
    dbc.Row([
        dbc.Col(html.H1("Global Fraud Analytics", className="text-start my-4", 
                        style={"font-weight": "700", "padding-left": "15px"}), width=12)
    ]),

    # Main Content Row
    dbc.Row([
        
        # LEFT COLUMN: Stats & Controls (Width: 3)
        dbc.Col([
            # Stats Card 1: Total Volume
            dbc.Card([
                dbc.CardBody([
                    html.H6("Total Transactions", className="text-muted"),
                    html.H3(f"{len(df)}", style={"color": "#00d4ff"})
                ])
            ], style={"background": "#111", "border": "1px solid #333", "margin-bottom": "15px"}),

            # Stats Card 2: Median Transaction Amount
            dbc.Card([
                dbc.CardBody([
                    html.H6("Median Transaction Amount", className="text-muted"),
                    html.H3(f"{(df['is_fraud'].median()*100):.1f}%", style={"color": "#00d4ff"})
                ])
            ], style={"background": "#111", "border": "1px solid #333", "margin-bottom": "15px"}),

            # Stats Card 3: Fraud Rate
            dbc.Card([
                dbc.CardBody([
                    html.H6("Avg. Fraud Rate", className="text-muted"),
                    html.H3(f"{(df['is_fraud'].mean()*100):.1f}%", style={"color": "#ff0055"})
                ])
            ], style={"background": "#111", "border": "1px solid #333", "margin-bottom": "30px"}),

            # Metric Toggle
            html.Div([
                html.Label("View Mode", style={"margin-bottom": "10px", "opacity": "0.7", "font-weight": "bold"}),
                dcc.RadioItems(
                    id='map-metric-selector',
                    options=[
                        {'label': ' Transaction Volume', 'value': 'volume'},
                        {'label': ' Fraud Probability', 'value': 'rate'}
                    ],
                    value='volume',
                    labelStyle={'display': 'block', 'margin-bottom': '10px'}, # Stacked vertically
                    inputStyle={"margin-right": "10px"},
                    style={"padding": "20px", "border-radius": "8px", "background": "#111", "border": "1px solid #222"}
                ),
            ])
        ], width=12, lg=3), # Takes 3/12 width on large screens

        # RIGHT COLUMN: The Map (Width: 9)
        dbc.Col([
            dbc.Card([
                dcc.Graph(
                    id='interactive-map', 
                    config={'displayModeBar': False},
                    style={"height": "70vh"} # Higher map height for better visibility
                )
            ], style={"background": "#111", "border": "1px solid #333", "padding": "10px"})
        ], width=12, lg=9)
    ], className="g-4") # 'g-4' adds a nice gap between columns
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