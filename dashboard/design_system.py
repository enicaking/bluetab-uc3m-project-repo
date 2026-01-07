"""
Professional Financial Dashboard Design System
Shared constants and styling across all pages
"""

# ============================================
# COLOR PALETTE (Financial Professional)
# ============================================

# Backgrounds
BG_PRIMARY = "#0a0a0a"      # Deep black
BG_SECONDARY = "#141414"    # Dark gray
BG_TERTIARY = "#1a1a1a"     # Lighter dark gray
BG_CARD = "#141414"         # Card background

# Borders & Dividers
BORDER_PRIMARY = "#2a2a2a"
BORDER_SECONDARY = "#3a3a3a"
BORDER_HOVER = "#4a4a4a"

# Text Colors
TEXT_PRIMARY = "#FFFFFF"
TEXT_SECONDARY = "#CCCCCC"
TEXT_MUTED = "#888888"
TEXT_DISABLED = "#666666"

# Accent Colors (Financial)
ACCENT_PRIMARY = "#00d4ff"      # Cyan (primary actions, positive)
ACCENT_SECONDARY = "#0099cc"    # Darker cyan
ACCENT_DANGER = "#ff0055"       # Red (fraud, alerts)
ACCENT_SUCCESS = "#00ff88"      # Green (positive metrics)
ACCENT_WARNING = "#ffaa00"      # Orange (warnings)
ACCENT_INFO = "#00d4ff"         # Info blue

# Gradient Definitions
GRADIENT_PRIMARY = "linear-gradient(135deg, #00d4ff 0%, #0099cc 100%)"
GRADIENT_DANGER = "linear-gradient(135deg, #ff0055 0%, #cc0033 100%)"
GRADIENT_SUCCESS = "linear-gradient(135deg, #00ff88 0%, #00cc66 100%)"
GRADIENT_CARD = "linear-gradient(135deg, #141414 0%, #1a1a1a 100%)"
GRADIENT_NAVBAR = "linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%)"
GRADIENT_TEXT = "linear-gradient(135deg, #ffffff 0%, #cccccc 100%)"

# ============================================
# TYPOGRAPHY
# ============================================

FONT_FAMILY = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
FONT_WEIGHT_LIGHT = 300
FONT_WEIGHT_REGULAR = 400
FONT_WEIGHT_MEDIUM = 500
FONT_WEIGHT_SEMIBOLD = 600
FONT_WEIGHT_BOLD = 700
FONT_WEIGHT_EXTRABOLD = 800

# ============================================
# SPACING SYSTEM
# ============================================

SPACING_XS = "0.25rem"   # 4px
SPACING_SM = "0.5rem"    # 8px
SPACING_MD = "1rem"      # 16px
SPACING_LG = "1.5rem"    # 24px
SPACING_XL = "2rem"      # 32px
SPACING_XXL = "3rem"      # 48px

# ============================================
# BORDER RADIUS
# ============================================

RADIUS_SM = "6px"
RADIUS_MD = "8px"
RADIUS_LG = "12px"
RADIUS_XL = "16px"

# ============================================
# SHADOWS
# ============================================

SHADOW_SM = "0 2px 4px rgba(0, 0, 0, 0.3)"
SHADOW_MD = "0 4px 12px rgba(0, 0, 0, 0.4)"
SHADOW_LG = "0 8px 24px rgba(0, 0, 0, 0.5)"
SHADOW_GLOW = "0 0 20px rgba(0, 212, 255, 0.3)"

# ============================================
# STYLE DICTIONARIES
# ============================================

DARK_STYLE = {
    "backgroundColor": BG_PRIMARY,
    "color": TEXT_PRIMARY,
    "fontFamily": FONT_FAMILY,
    "minHeight": "100vh",
}

CARD_STYLE = {
    "background": GRADIENT_CARD,
    "border": f"1px solid {BORDER_PRIMARY}",
    "borderRadius": RADIUS_LG,
    "boxShadow": SHADOW_MD,
    "transition": "all 0.3s ease",
    "overflow": "hidden",
}

CARD_STYLE_HOVER = {
    **CARD_STYLE,
    "transform": "translateY(-2px)",
    "boxShadow": SHADOW_LG,
    "borderColor": BORDER_SECONDARY,
}

PANEL_STYLE = {
    "background": GRADIENT_CARD,
    "border": f"1px solid {BORDER_PRIMARY}",
    "borderRadius": RADIUS_LG,
    "boxShadow": SHADOW_MD,
}

KPI_CARD_STYLE = {
    **CARD_STYLE,
    "padding": SPACING_LG,
    "position": "relative",
}

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_gradient_text_style(color_start="#FFFFFF", color_end="#CCCCCC"):
    """Generate gradient text style"""
    return {
        "background": f"linear-gradient(135deg, {color_start} 0%, {color_end} 100%)",
        "-webkit-background-clip": "text",
        "-webkit-text-fill-color": "transparent",
        "background-clip": "text",
    }

def get_accent_color(metric_type="primary"):
    """Get accent color based on metric type"""
    colors = {
        "primary": ACCENT_PRIMARY,
        "danger": ACCENT_DANGER,
        "success": ACCENT_SUCCESS,
        "warning": ACCENT_WARNING,
        "info": ACCENT_INFO,
    }
    return colors.get(metric_type, ACCENT_PRIMARY)


