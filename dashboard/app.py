import dash
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

# financial dashboard theme
app = Dash(
    __name__, 
    use_pages=True, 
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True
)

# Global Styling Dictionary
DARK_STYLE = {
    "background-color": "#0a0a0a",
    "color": "#FFFFFF",
    "font-family": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    "min-height": "100vh",
}

app.layout = html.Div(style=DARK_STYLE, children=[
    # Location component to track current path
    dcc.Location(id="url", refresh=False),
    
    # External Font Import
    html.Link(
        href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap", 
        rel="stylesheet"
    ),
    
    # Custom CSS
    html.Link(rel="stylesheet", href="/assets/styles.css"),

    # Professional Navigation Header
    dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand(
                [
                    html.Span("🔒 ", style={"fontSize": "1.5rem", "marginRight": "0.5rem"}),
                    html.Span("Fraud Analytics", style={"fontWeight": "700", "letterSpacing": "-0.5px"}),
                    html.Small(" | UC3M-Bluetab", style={"fontSize": "0.75rem", "opacity": "0.7", "marginLeft": "0.5rem"})
                ],
                href="/",
                className="navbar-brand",
                style={"fontSize": "1.25rem"}
            ),
            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Collapse(
                dbc.Nav(
                    id="nav-links",
                    className="ms-auto",
                    navbar=True
                ),
                id="navbar-collapse",
                navbar=True,
            ),
        ], fluid=True),
        color="dark",
        dark=True,
        className="navbar",
        style={"boxShadow": "0 2px 8px rgba(0,0,0,0.3)"}
    ),
 
    # Main content area
    dbc.Container([
        dash.page_container
    ], fluid=True, style={"padding": "2rem", "maxWidth": "1920px"})
])

# Callback to create nav links and highlight active page
@app.callback(
    Output("nav-links", "children"),
    Input("url", "pathname")
)
def update_nav_links(pathname):
    """Create nav links and highlight the active page"""
    # Sort pages by order, handling None values
    def get_order(page):
        order = page.get('order')
        return order if order is not None else 999
    
    pages = sorted(dash.page_registry.values(), key=get_order)
    
    nav_items = []
    for page in pages:
        page_path = page.get("relative_path", "")
        page_name = page.get('name', 'Unknown')
        
        # Check if current path matches this page
        is_active = pathname == page_path or (page_path == "/" and pathname == "/")
        
        if is_active:
            className = "nav-link nav-link-active"
        else:
            className = "nav-link"
        
        nav_items.append(
            dbc.NavItem(
                dcc.Link(
                    page_name, 
                    href=page_path, 
                    className=className,
                    style={"textDecoration": "none"}
                )
            )
        )
    
    return nav_items

if __name__ == '__main__':
    app.run(debug=True)