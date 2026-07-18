"""
Thème visuel "Azure Profond" pour le dashboard.
Injecte du CSS personnalisé et fournit des composants HTML réutilisables
(cartes KPI, badges de statut, pastilles de notification) sans emoji :
les icônes proviennent de Bootstrap Icons via streamlit-option-menu,
et les indicateurs visuels utilisent des formes géométriques simples (CSS).
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Palette "Azure Profond"
# ---------------------------------------------------------------------------
PALETTE = {
    "dark": {
        "bg": "#061426",
        "bg_secondary": "#0B1F38",
        "panel": "#0E2947",
        "panel_hover": "#123458",
        "border": "#1C3A5E",
        "text_primary": "#EAF2FB",
        "text_secondary": "#93A9C4",
        "accent": "#1B6FC9",
        "accent_light": "#3E8FE0",
        "accent_soft": "#123A63",
        "success": "#1FA36A",
        "warning": "#D98E2E",
        "critical": "#C74B4B",
    },
    "light": {
        "bg": "#F4F8FC",
        "bg_secondary": "#FFFFFF",
        "panel": "#FFFFFF",
        "panel_hover": "#EAF1FB",
        "border": "#D7E3F0",
        "text_primary": "#0B1F38",
        "text_secondary": "#4C6079",
        "accent": "#0B5FB3",
        "accent_light": "#1B6FC9",
        "accent_soft": "#E4EEFB",
        "success": "#1B8A56",
        "warning": "#B8791F",
        "critical": "#B23A3A",
    },
}


def inject_css(mode: str = "dark"):
    p = PALETTE[mode]
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        .stApp {{
            background: {p['bg']};
            color: {p['text_primary']};
        }}

        section[data-testid="stSidebar"] {{
            background: {p['bg_secondary']};
            border-right: 1px solid {p['border']};
        }}

        section[data-testid="stSidebar"] .block-container {{
            padding-top: 1.2rem;
        }}

        h1, h2, h3, h4 {{
            color: {p['text_primary']} !important;
            font-weight: 700 !important;
            letter-spacing: -0.01em;
        }}

        p, span, label, .stMarkdown {{
            color: {p['text_secondary']};
        }}

        /* ---- Header de page ---- */
        .platform-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1.1rem 1.6rem;
            background: linear-gradient(135deg, {p['panel']} 0%, {p['bg_secondary']} 100%);
            border: 1px solid {p['border']};
            border-radius: 14px;
            margin-bottom: 1.6rem;
        }}
        .platform-header .title {{
            font-size: 1.5rem;
            font-weight: 800;
            color: {p['text_primary']};
        }}
        .platform-header .subtitle {{
            font-size: 0.85rem;
            color: {p['text_secondary']};
            margin-top: 2px;
        }}

        /* ---- Cartes KPI ---- */
        .kpi-card {{
            background: {p['panel']};
            border: 1px solid {p['border']};
            border-radius: 14px;
            padding: 1.15rem 1.3rem;
            transition: all 0.18s ease;
            height: 100%;
        }}
        .kpi-card:hover {{
            background: {p['panel_hover']};
            border-color: {p['accent']};
            transform: translateY(-2px);
        }}
        .kpi-label {{
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: {p['text_secondary']};
            font-weight: 600;
            margin-bottom: 0.5rem;
        }}
        .kpi-value {{
            font-size: 1.85rem;
            font-weight: 800;
            color: {p['text_primary']};
            line-height: 1.1;
        }}
        .kpi-sub {{
            font-size: 0.78rem;
            color: {p['text_secondary']};
            margin-top: 0.35rem;
        }}
        .kpi-accent-bar {{
            height: 3px;
            width: 36px;
            border-radius: 3px;
            background: {p['accent']};
            margin-bottom: 0.7rem;
        }}

        /* ---- Cartes module (accueil) ---- */
        .module-card {{
            background: {p['panel']};
            border: 1px solid {p['border']};
            border-radius: 14px;
            padding: 1.3rem 1.4rem;
            transition: all 0.18s ease;
            height: 100%;
        }}
        .module-card:hover {{
            border-color: {p['accent']};
            background: {p['panel_hover']};
        }}
        .module-title {{
            font-size: 1.02rem;
            font-weight: 700;
            color: {p['text_primary']};
        }}
        .module-metric {{
            font-size: 1.5rem;
            font-weight: 800;
            color: {p['accent_light']};
            margin: 0.4rem 0 0.2rem 0;
        }}
        .module-caption {{
            font-size: 0.8rem;
            color: {p['text_secondary']};
        }}

        /* ---- Badges de statut ---- */
        .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.74rem;
            font-weight: 600;
            padding: 3px 10px 3px 8px;
            border-radius: 999px;
        }}
        .status-dot {{
            width: 7px; height: 7px; border-radius: 50%;
        }}
        .status-ok {{ background: {p['accent_soft']}; color: {p['accent_light']}; }}
        .status-ok .status-dot {{ background: {p['success']}; }}
        .status-warning {{ background: {p['accent_soft']}; color: {p['warning']}; }}
        .status-warning .status-dot {{ background: {p['warning']}; }}
        .status-critical {{ background: {p['accent_soft']}; color: {p['critical']}; }}
        .status-critical .status-dot {{ background: {p['critical']}; }}

        /* ---- Notification list ---- */
        .notif-item {{
            display: flex;
            gap: 10px;
            padding: 0.7rem 0.85rem;
            border-radius: 10px;
            border: 1px solid {p['border']};
            background: {p['panel']};
            margin-bottom: 0.55rem;
        }}
        .notif-bar {{
            width: 3px;
            border-radius: 3px;
            flex-shrink: 0;
        }}
        .notif-critical .notif-bar {{ background: {p['critical']}; }}
        .notif-attention .notif-bar {{ background: {p['warning']}; }}
        .notif-text {{ font-size: 0.85rem; color: {p['text_primary']}; }}
        .notif-time {{ font-size: 0.72rem; color: {p['text_secondary']}; margin-top: 2px; }}

        /* ---- Section container ---- */
        .section-panel {{
            background: {p['panel']};
            border: 1px solid {p['border']};
            border-radius: 14px;
            padding: 1.4rem 1.5rem;
            margin-bottom: 1.3rem;
        }}
        .section-eyebrow {{
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
            color: {p['accent_light']};
            margin-bottom: 0.3rem;
        }}

        /* ---- Streamlit natives ---- */
        .stButton > button {{
            background: {p['accent']};
            color: white;
            border: none;
            border-radius: 9px;
            font-weight: 600;
            padding: 0.5rem 1.1rem;
            transition: all 0.15s ease;
        }}
        .stButton > button:hover {{
            background: {p['accent_light']};
            box-shadow: 0 4px 14px rgba(27, 111, 201, 0.35);
        }}
        div[data-baseweb="tab-list"] {{ gap: 4px; }}

        .stDataFrame {{ border-radius: 12px; overflow: hidden; }}

        [data-testid="stMetricValue"] {{ color: {p['text_primary']}; }}

        hr {{ border-color: {p['border']}; }}
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Composants réutilisables
# ---------------------------------------------------------------------------
def page_header(title: str, subtitle: str, right_html: str = ""):
    st.markdown(f"""
    <div class="platform-header">
        <div>
            <div class="title">{title}</div>
            <div class="subtitle">{subtitle}</div>
        </div>
        <div>{right_html}</div>
    </div>
    """, unsafe_allow_html=True)


def kpi_card(label: str, value: str, sub: str = ""):
    return f"""
    <div class="kpi-card">
        <div class="kpi-accent-bar"></div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """


def status_pill(text: str, level: str = "ok"):
    cls = {"ok": "status-ok", "warning": "status-warning", "critical": "status-critical"}[level]
    return f'<span class="status-pill {cls}"><span class="status-dot"></span>{text}</span>'


def module_card(title: str, metric: str, caption: str, status_html: str):
    return f"""
    <div class="module-card">
        {status_html}
        <div class="module-title" style="margin-top:0.6rem;">{title}</div>
        <div class="module-metric">{metric}</div>
        <div class="module-caption">{caption}</div>
    </div>
    """


def notification_item(message: str, time_str: str, severity: str = "attention"):
    cls = "notif-critical" if severity == "critique" else "notif-attention"
    return f"""
    <div class="notif-item {cls}">
        <div class="notif-bar"></div>
        <div>
            <div class="notif-text">{message}</div>
            <div class="notif-time">{time_str}</div>
        </div>
    </div>
    """
