import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

# external_stylesheets uses the CYBORG theme for a dark baseline
app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.CYBORG])

# Global Styling Dictionary
DARK_STYLE = {
    "background-color": "#000000",
    "color": "#FFFFFF",
    "font-family": "'Inter', sans-serif",
    "min-height": "100vh",  # Ensures the whole screen stays black
}

app.layout = html.Div(style=DARK_STYLE, children=[
    # External Font Import
    html.Link(href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700&display=swap", rel="stylesheet"),

    # Navigation Header
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dcc.Link(page['name'], href=page["relative_path"], className="nav-link"))
            for page in dash.page_registry.values()
        ],
        brand="UC3M-Bluetab Fraud Analytics Dashboard",
        brand_href="#",
        color="black",
        dark=True,
        className="mb-4",
        style={"border-bottom": "1px solid #333"}
    ),
 
    # This is where the pages (like data_explore.py) will be injected
    dbc.Container([
        dash.page_container
    ], fluid=True)
])

if __name__ == '__main__':
    app.run(debug=True)