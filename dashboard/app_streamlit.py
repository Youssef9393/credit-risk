"""
Dashboard Streamlit - Plateforme Intelligente de Risque Crédit
==================================================================
Lancer avec :
    streamlit run dashboard/app_streamlit.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "raw"

st.set_page_config(page_title="Plateforme Risque Crédit & Fraude", layout="wide", page_icon="▮")

# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
st.session_state.setdefault("theme", "dark")
st.session_state.setdefault("nav_page", "Tableau de bord")

PAGES = ["Tableau de bord", "Credit Scoring", "Détection de fraude",
         "Segmentation clients", "Sentiment financier", "Prévision boursière"]

# ---------------------------------------------------------------------------
# ICONS (inline SVG, no emoji — one distinct glyph per business domain)
# ---------------------------------------------------------------------------
ICONS = {
    "dashboard": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="3" y="3" width="7" height="9" rx="1"/><rect x="14" y="3" width="7" height="5" rx="1"/><rect x="14" y="12" width="7" height="9" rx="1"/><rect x="3" y="16" width="7" height="5" rx="1"/></svg>',
    "credit": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 10h18"/><path d="M7 15h4"/></svg>',
    "fraud": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M12 3l7 3v6c0 4.5-3 7.5-7 9-4-1.5-7-4.5-7-9V6l7-3z"/><path d="M9.5 12l1.8 1.8 3.2-3.6"/></svg>',
    "cluster": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="8" cy="8" r="3"/><circle cx="17" cy="7" r="2.4"/><circle cx="16" cy="17" r="3"/><circle cx="7" cy="17" r="2.2"/></svg>',
    "nlp": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M4 5h16v10H9l-4 4V5z"/><path d="M8 9h8M8 12h5"/></svg>',
    "stock": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M3 17l6-6 4 4 8-9"/><path d="M15 6h6v6"/></svg>',
    "bell": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M6 9a6 6 0 1 1 12 0c0 5 2 6 2 6H4s2-1 2-6z"/><path d="M10 20a2 2 0 0 0 4 0"/></svg>',
}

# ---------------------------------------------------------------------------
# THEME TOKENS — azur profond (dark) / azur clair (light)
# ---------------------------------------------------------------------------
THEME = {
    "dark": dict(bg="#061B33", panel="#0B2545", panel_alt="#0F2E52", line="#1E3D63",
                 text_hi="#EAF2FB", text_lo="#8FB0D6", accent="#2EA0FF", accent_ink="#04121F",
                 ok="#3FCB99", warn="#E7B24F", bad="#F1667C"),
    "light": dict(bg="#F2F7FC", panel="#FFFFFF", panel_alt="#E9F2FB", line="#D6E4F2",
                  text_hi="#0B2545", text_lo="#5D7C9E", accent="#0B76D1", accent_ink="#FFFFFF",
                  ok="#1F8F6D", warn="#B4740B", bad="#C23B4B"),
}
t = THEME[st.session_state.theme]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"], .stMarkdown, p, span {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background: {t['bg']}; color: {t['text_hi']}; }}
section.main > div {{ padding-top: 1rem; }}

/* Header */
.platform-header {{
    display:flex; justify-content:space-between; align-items:center;
    border-bottom:1px solid {t['line']}; padding-bottom:16px; margin-bottom:22px; flex-wrap:wrap; gap:12px;
}}
.platform-title {{ font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.7rem; letter-spacing:-0.02em; margin:0; color:{t['text_hi']}; }}
.platform-sub {{ font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:{t['text_lo']}; letter-spacing:0.05em; text-transform:uppercase; margin-top:5px; }}

/* Sidebar / nav */
[data-testid="stSidebar"] {{ background:{t['panel']}; border-right:1px solid {t['line']}; }}
[data-testid="stSidebar"] * {{ color:{t['text_hi']}; }}
.sb-title {{ font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.05rem; margin-bottom:2px; }}
.sb-sub {{ font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:{t['text_lo']}; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:18px; }}
[data-testid="stSidebar"] .stRadio > label {{ display:none; }}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] {{ gap:2px; }}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] label {{
    background:transparent; border-radius:4px; padding:9px 10px; font-family:'IBM Plex Mono',monospace;
    font-size:0.82rem; letter-spacing:0.01em; border:1px solid transparent;
}}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {{ background:{t['panel_alt']}; }}
[data-testid="stSidebar"] .stRadio [aria-checked="true"] {{
    background:{t['panel_alt']} !important; border:1px solid {t['line']} !important; color:{t['accent']} !important;
}}

/* KPI / module cards */
.kpi-card {{
    background:{t['panel']}; border:1px solid {t['line']}; border-radius:8px; padding:18px 20px;
    transition:border-color .15s ease, transform .15s ease;
}}
.kpi-card:hover {{ border-color:{t['accent']}; transform:translateY(-2px); }}
.kpi-top {{ display:flex; align-items:center; gap:10px; color:{t['text_lo']}; }}
.kpi-name {{ font-family:'IBM Plex Mono',monospace; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em; }}
.kpi-metric {{ font-family:'Space Grotesk',sans-serif; font-size:1.18rem; font-weight:600; margin:12px 0 4px 0; color:{t['text_hi']}; }}
.status-line {{ display:flex; align-items:center; gap:7px; font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:{t['text_lo']}; }}
.dot {{ width:6px; height:6px; border-radius:50%; flex-shrink:0; }}
.dot-ok {{ background:{t['ok']}; box-shadow:0 0 6px {t['ok']}; }}
.dot-warn {{ background:{t['warn']}; box-shadow:0 0 6px {t['warn']}; }}
.dot-bad {{ background:{t['bad']}; box-shadow:0 0 6px {t['bad']}; }}

/* Résumé bar inside module pages */
.summary-bar {{
    display:flex; align-items:center; gap:10px; background:{t['panel']}; border:1px solid {t['line']};
    border-left:3px solid {t['accent']}; border-radius:6px; padding:12px 16px; margin-bottom:22px;
    font-family:'IBM Plex Mono',monospace; font-size:0.82rem; color:{t['text_hi']};
}}

.panel-label {{
    font-family:'IBM Plex Mono',monospace; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em;
    color:{t['text_lo']}; margin:4px 0 14px 0; border-bottom:1px solid {t['line']}; padding-bottom:8px;
}}

div[data-testid="stVerticalBlockBorderWrapper"] {{ background:{t['panel']}; border:1px solid {t['line']} !important; border-radius:8px; }}

.stButton > button {{
    font-family:'IBM Plex Mono',monospace; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em;
    background:{t['accent']}; color:{t['accent_ink']}; border:none; border-radius:4px; padding:9px 18px; font-weight:600;
}}
.stButton > button:hover {{ opacity:0.87; color:{t['accent_ink']}; }}
.stButton > button:focus:not(:active) {{ color:{t['accent_ink']}; }}
button[kind="secondary"] {{ background:transparent !important; color:{t['accent']} !important; border:1px solid {t['line']} !important; }}

.stNumberInput input, .stTextArea textarea {{
    font-family:'IBM Plex Mono',monospace !important; background:{t['panel_alt']} !important;
    color:{t['text_hi']} !important; border:1px solid {t['line']} !important; border-radius:4px !important;
}}
label, .stSlider label, .stNumberInput label, .stTextArea label {{
    font-family:'IBM Plex Mono',monospace !important; font-size:0.72rem !important; text-transform:uppercase;
    letter-spacing:0.03em; color:{t['text_lo']} !important;
}}
[data-testid="stMetricValue"] {{ font-family:'IBM Plex Mono',monospace; color:{t['accent']}; }}
[data-testid="stMetricLabel"] {{ font-family:'IBM Plex Mono',monospace; text-transform:uppercase; font-size:0.7rem; color:{t['text_lo']}; }}
.stAlert {{ border-radius:6px; font-family:'Inter',sans-serif; border:1px solid {t['line']}; }}
hr {{ border-color:{t['line']} !important; }}
[data-testid="stDataFrame"] {{ border:1px solid {t['line']}; border-radius:6px; }}
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=t["panel"], plot_bgcolor=t["panel"],
        font=dict(family="IBM Plex Mono, monospace", color=t["text_hi"], size=12),
        colorway=[t["accent"], t["warn"], t["bad"], t["text_lo"], t["ok"]],
        xaxis=dict(gridcolor=t["line"], zerolinecolor=t["line"]),
        yaxis=dict(gridcolor=t["line"], zerolinecolor=t["line"]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
)

# ---------------------------------------------------------------------------
# BUSINESS KPIs — computed from real artifacts where available
# ---------------------------------------------------------------------------
credit_ready = (MODEL_DIR / "credit_scoring_model.joblib").exists()
fraud_ready = (MODEL_DIR / "fraud_xgboost.joblib").exists()
nlp_ready = (MODEL_DIR / "nlp_sentiment_baseline.joblib").exists()

seg_path = BASE_DIR / "data" / "customer_segments.csv"
stock_path = DATA_DIR / "stock_prices.csv"

if seg_path.exists():
    _df_seg = pd.read_csv(seg_path)
    cluster_metric = f"{_df_seg['segment_name'].nunique()} segments actifs"
    cluster_sub = f"{len(_df_seg)} clients classés"
else:
    cluster_metric, cluster_sub = "Segmentation indisponible", "Données non chargées"

if stock_path.exists():
    _df_stock = pd.read_csv(stock_path, parse_dates=["Date"])
    _last, _prev = _df_stock["Close"].iloc[-1], _df_stock["Close"].iloc[-2]
    _pct = (_last - _prev) / _prev * 100
    _arrow = "▲" if _pct >= 0 else "▼"
    lstm_metric = f"Tendance {_arrow} {_pct:+.2f}%"
    lstm_sub = f"Dernière clôture : {_last:,.2f}"
else:
    _pct = 0.0
    lstm_metric, lstm_sub = "Cours indisponible", "Données non chargées"

MODULES = [
    dict(key="credit", page="Credit Scoring", label="Credit Scoring", ready=credit_ready,
         metric="Notation de dossiers", sub="Modèle prêt à évaluer" if credit_ready else "Modèle non entraîné",
         action="Évaluer un dossier"),
    dict(key="fraud", page="Détection de fraude", label="Détection de fraude", ready=fraud_ready,
         metric="Surveillance transactionnelle", sub="Contrôle en temps réel actif" if fraud_ready else "Modèle non entraîné",
         action="Contrôler une transaction"),
    dict(key="cluster", page="Segmentation clients", label="Segmentation clients", ready=seg_path.exists(),
         metric=cluster_metric, sub=cluster_sub, action="Explorer les segments"),
    dict(key="nlp", page="Sentiment financier", label="Sentiment financier", ready=nlp_ready,
         metric="Lecture des actualités", sub="Moteur prêt à analyser" if nlp_ready else "Modèle non entraîné",
         action="Analyser une actualité"),
    dict(key="stock", page="Prévision boursière", label="Prévision boursière", ready=stock_path.exists(),
         metric=lstm_metric, sub=lstm_sub, action="Voir la prévision"),
]

NOTIFICATIONS = []
for m in MODULES:
    if not m["ready"]:
        NOTIFICATIONS.append(("warn", f"{m['label']} — {m['sub']}"))
if stock_path.exists() and _pct <= -2:
    NOTIFICATIONS.append(("bad", f"Prévision boursière — repli marqué de {_pct:.2f}% sur la dernière séance"))
if not NOTIFICATIONS:
    NOTIFICATIONS.append(("ok", "Aucune alerte critique en attente"))

# ---------------------------------------------------------------------------
# SIDEBAR — navigation unique + statut métier + apparence
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sb-title">Plateforme Risque</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-sub">Crédit · Fraude · Marché</div>', unsafe_allow_html=True)

    light_mode = st.toggle("Mode clair", value=(st.session_state.theme == "light"))
    new_theme = "light" if light_mode else "dark"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

    st.session_state.nav_page = st.radio(
        "Navigation", PAGES, index=PAGES.index(st.session_state.nav_page), key="nav_radio",
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sb-sub">Statut des modules</div>', unsafe_allow_html=True)
    for m in MODULES:
        dot = "dot-ok" if m["ready"] else "dot-warn"
        st.markdown(
            f'<div class="status-line"><span class="dot {dot}"></span>{m["label"]} — {m["sub"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

page = st.session_state.nav_page

# ---------------------------------------------------------------------------
# HEADER (with notification center)
# ---------------------------------------------------------------------------
n_alerts = sum(1 for lvl, _ in NOTIFICATIONS if lvl != "ok")
head_l, head_r = st.columns([5, 1])
with head_l:
    st.markdown(f"""
    <div class="platform-header" style="border-bottom:none; margin-bottom:0; padding-bottom:0;">
        <div>
            <p class="platform-title">{page}</p>
            <p class="platform-sub">Credit Scoring · Détection de fraude · Segmentation client · NLP financier · Prévision boursière</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
with head_r:
    try:
        with st.popover(f"Alertes ({n_alerts})" if n_alerts else "Alertes"):
            for lvl, msg in NOTIFICATIONS:
                dot = {"ok": "dot-ok", "warn": "dot-warn", "bad": "dot-bad"}[lvl]
                st.markdown(f'<div class="status-line"><span class="dot {dot}"></span>{msg}</div>',
                            unsafe_allow_html=True)
    except AttributeError:
        with st.expander(f"Alertes ({n_alerts})"):
            for lvl, msg in NOTIFICATIONS:
                dot = {"ok": "dot-ok", "warn": "dot-warn", "bad": "dot-bad"}[lvl]
                st.markdown(f'<div class="status-line"><span class="dot {dot}"></span>{msg}</div>',
                            unsafe_allow_html=True)
st.markdown(f"<div style='border-bottom:1px solid {t['line']}; margin:14px 0 24px 0;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PAGE : TABLEAU DE BORD
# ---------------------------------------------------------------------------
if page == "Tableau de bord":
    st.markdown('<div class="panel-label">Vue d\'ensemble — indicateurs métiers</div>', unsafe_allow_html=True)
    cols = st.columns(len(MODULES))
    for col, m in zip(cols, MODULES):
        with col:
            dot = "dot-ok" if m["ready"] else "dot-warn"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-top">{ICONS[m['key']]}<span class="kpi-name">{m['label']}</span></div>
                <div class="kpi-metric">{m['metric']}</div>
                <div class="status-line"><span class="dot {dot}"></span>{m['sub']}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(m["action"], key=f"go_{m['key']}", use_container_width=True):
                st.session_state.nav_page = m["page"]
                st.rerun()

    st.write("")
    st.markdown('<div class="panel-label">Tendance boursière récente</div>', unsafe_allow_html=True)
    if stock_path.exists():
        fig = px.line(_df_stock.tail(120), x="Date", y="Close")
        fig.update_traces(line_color=t["accent"])
        fig.update_layout(template=PLOTLY_TEMPLATE, height=300, margin=dict(t=10, b=30, l=40, r=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Données boursières non chargées. Lancez `python src/data_generation.py`.")

# ---------------------------------------------------------------------------
# PAGE : CREDIT SCORING
# ---------------------------------------------------------------------------
elif page == "Credit Scoring":
    st.markdown(f'<div class="summary-bar">{ICONS["credit"]}&nbsp; {"Modèle de notation actif — prêt à évaluer un dossier" if credit_ready else "Modèle de notation non entraîné"}</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel-label">Profil client — paramètres d\'entrée</div>', unsafe_allow_html=True)
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Âge", 18, 100, 35)
            income = st.number_input("Revenu mensuel ($)", 0, 100000, 4500)
            debt_ratio = st.slider("Taux d'endettement", 0.0, 1.0, 0.3)
        with col2:
            num_credit_lines = st.number_input("Nombre de lignes de crédit", 0, 40, 5)
            num_dependents = st.number_input("Personnes à charge", 0, 10, 1)
            revolving_util = st.slider("Taux d'utilisation renouvelable", 0.0, 1.5, 0.4)
        with col3:
            late_30_59 = st.number_input("Retards 30-59j", 0, 20, 0)
            late_60_89 = st.number_input("Retards 60-89j", 0, 20, 0)
            late_90 = st.number_input("Retards 90j+", 0, 20, 0)

    st.write("")
    if st.button("Calculer le score de risque", type="primary"):
        try:
            from credit_scoring import predict_default_probability
            client = {
                "age": age, "MonthlyIncome": income, "DebtRatio": debt_ratio,
                "NumberOfOpenCreditLinesAndLoans": num_credit_lines,
                "NumberOfDependents": num_dependents,
                "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
                "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
                "NumberOfTimes90DaysLate": late_90,
                "RevolvingUtilizationOfUnsecuredLines": revolving_util,
            }
            proba = predict_default_probability(client)
            risk_pct = proba * 100

            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=risk_pct,
                number={"suffix": "%", "font": {"family": "IBM Plex Mono, monospace", "size": 40}},
                title={"text": "PROBABILITÉ DE DÉFAUT", "font": {"family": "IBM Plex Mono, monospace", "size": 13}},
                gauge={"axis": {"range": [0, 100], "tickcolor": t["text_lo"]},
                       "bar": {"color": t["bad"] if risk_pct > 50 else t["accent"]},
                       "bgcolor": t["panel_alt"], "borderwidth": 1, "bordercolor": t["line"],
                       "steps": [{"range": [0, 30], "color": t["panel_alt"]},
                                 {"range": [30, 60], "color": t["panel_alt"]},
                                 {"range": [60, 100], "color": t["panel_alt"]}]},
            ))
            fig.update_layout(template=PLOTLY_TEMPLATE, height=320, margin=dict(t=60, b=10, l=30, r=30))
            st.plotly_chart(fig, use_container_width=True)

            if risk_pct < 30:
                st.success("Risque faible — crédit recommandé")
            elif risk_pct < 60:
                st.warning("Risque modéré — analyse complémentaire recommandée")
            else:
                st.error("Risque élevé — crédit déconseillé")
        except FileNotFoundError:
            st.error("Modèle non entraîné. Lancez d'abord : `python src/credit_scoring.py`")

# ---------------------------------------------------------------------------
# PAGE : DÉTECTION DE FRAUDE
# ---------------------------------------------------------------------------
elif page == "Détection de fraude":
    st.markdown(f'<div class="summary-bar">{ICONS["fraud"]}&nbsp; {"Surveillance transactionnelle active" if fraud_ready else "Surveillance indisponible — modèle non entraîné"}</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel-label">Analyse d\'une transaction</div>', unsafe_allow_html=True)
    st.info("Renseignez le montant et l'heure ; les features V1-V28 sont simulées aléatoirement "
            "(dans un vrai déploiement, elles proviennent du pipeline de features en amont).")

    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            amount = st.number_input("Montant de la transaction ($)", 0.0, 50000.0, 120.0)
        with c2:
            hour = st.slider("Heure de la transaction (0-24h)", 0, 23, 14)

    st.write("")
    if st.button("Analyser la transaction", type="primary"):
        try:
            from feature_engineering import engineer_fraud_features
            model = joblib.load(MODEL_DIR / "fraud_xgboost.joblib")
            rng = np.random.default_rng(42)
            v_features = {f"V{i}": rng.normal(0, 1) for i in range(1, 29)}
            transaction = {**v_features, "Amount": amount, "Time": hour * 3600}
            df_tx = engineer_fraud_features(pd.DataFrame([transaction]))
            df_tx = df_tx[model.get_booster().feature_names]
            proba = model.predict_proba(df_tx)[:, 1][0]

            st.metric("Probabilité de fraude", f"{proba*100:.2f}%")
            if proba > 0.5:
                st.error("Transaction suspecte — vérification manuelle recommandée")
            else:
                st.success("Transaction jugée légitime")
        except FileNotFoundError:
            st.error("Modèle non entraîné. Lancez d'abord : `python src/fraud_detection.py`")

# ---------------------------------------------------------------------------
# PAGE : SEGMENTATION CLIENTS
# ---------------------------------------------------------------------------
elif page == "Segmentation clients":
    st.markdown(f'<div class="summary-bar">{ICONS["cluster"]}&nbsp; {cluster_metric} — {cluster_sub}</div>', unsafe_allow_html=True)

    if seg_path.exists():
        c1, c2 = st.columns([1, 2])
        with c1:
            with st.container(border=True):
                st.metric("Total clients", len(_df_seg))
                counts = _df_seg["segment_name"].value_counts()
                for seg, n in counts.items():
                    st.write(f"**{seg}** — {n} clients ({n/len(_df_seg)*100:.1f}%)")
        with c2:
            fig = px.pie(_df_seg, names="segment_name", title="Répartition des segments", hole=0.55)
            fig.update_layout(template=PLOTLY_TEMPLATE, height=340, margin=dict(t=50, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(
            _df_seg, x="annual_income", y="balance", color="segment_name",
            size="spending_score", hover_data=["age", "tenure_years"],
            title="Segments clients (revenu vs solde bancaire)",
        )
        fig2.update_layout(template=PLOTLY_TEMPLATE, height=440, margin=dict(t=50, b=30, l=40, r=20))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.error("Segmentation non calculée. Lancez d'abord : `python src/clustering.py`")

# ---------------------------------------------------------------------------
# PAGE : SENTIMENT FINANCIER
# ---------------------------------------------------------------------------
elif page == "Sentiment financier":
    st.markdown(f'<div class="summary-bar">{ICONS["nlp"]}&nbsp; {"Moteur de lecture des actualités prêt" if nlp_ready else "Moteur indisponible — modèle non entraîné"}</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel-label">Analyse de sentiment — actualité financière</div>', unsafe_allow_html=True)
    with st.container(border=True):
        text = st.text_area("Collez un titre ou un extrait d'actualité financière",
                             "Company Alpha Corp reports record profits this quarter")

    st.write("")
    if st.button("Analyser le sentiment", type="primary"):
        try:
            from nlp_sentiment import predict_sentiment_baseline
            sentiment = predict_sentiment_baseline(text)
            color = {"positive": t["ok"], "negative": t["bad"], "neutral": t["text_lo"]}[sentiment]
            st.markdown(
                f"""<div class="kpi-card" style="max-width:320px;">
                <div class="kpi-name">Sentiment détecté</div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:1.3rem; color:{color}; margin-top:8px;">
                {sentiment.upper()}</div></div>""",
                unsafe_allow_html=True,
            )
        except FileNotFoundError:
            st.error("Modèle non entraîné. Lancez d'abord : `python src/nlp_sentiment.py`")

# ---------------------------------------------------------------------------
# PAGE : PRÉVISION BOURSIÈRE
# ---------------------------------------------------------------------------
elif page == "Prévision boursière":
    st.markdown(f'<div class="summary-bar">{ICONS["stock"]}&nbsp; {lstm_metric} — {lstm_sub}</div>', unsafe_allow_html=True)

    if stock_path.exists():
        fig = px.line(_df_stock, x="Date", y="Close", title="Historique du cours de clôture")
        fig.update_traces(line_color=t["accent"])
        fig.update_layout(template=PLOTLY_TEMPLATE, height=360, margin=dict(t=50, b=30, l=40, r=20))
        st.plotly_chart(fig, use_container_width=True)

        with st.container(border=True):
            n_days = st.slider("Nombre de jours à prévoir", 1, 15, 5)
            generate_forecast = st.button("Générer la prévision", type="primary")

        if generate_forecast:
            try:
                import torch
                from time_series_lstm import LSTMForecaster, forecast_next_days, WINDOW_SIZE

                scaler = joblib.load(MODEL_DIR / "lstm_scaler.joblib")
                model = LSTMForecaster()
                model.load_state_dict(torch.load(MODEL_DIR / "lstm_stock_forecaster.pt"))

                last_window = _df_stock["Close"].values[-WINDOW_SIZE:]
                preds = forecast_next_days(model, scaler, last_window, n_days=n_days)

                future_dates = pd.bdate_range(
                    start=_df_stock["Date"].iloc[-1] + pd.Timedelta(days=1), periods=n_days
                )
                forecast_df = pd.DataFrame({"Date": future_dates, "Close": preds})

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=_df_stock["Date"].tail(60), y=_df_stock["Close"].tail(60),
                                           name="Historique", mode="lines", line=dict(color=t["text_lo"])))
                fig2.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Close"],
                                           name="Prévision", mode="lines+markers",
                                           line=dict(dash="dash", color=t["accent"])))
                fig2.update_layout(template=PLOTLY_TEMPLATE, height=400, margin=dict(t=30, b=30, l=40, r=20))
                st.plotly_chart(fig2, use_container_width=True)
                st.dataframe(forecast_df, use_container_width=True)
            except FileNotFoundError:
                st.error("Modèle LSTM non entraîné. Lancez d'abord : `python src/time_series_lstm.py`")
    else:
        st.error("Données boursières manquantes. Lancez d'abord : `python src/data_generation.py`")