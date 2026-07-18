"""
Centre de Décision - Plateforme de Risque Crédit, Fraude & Marchés
========================================================================
Lancer avec :
    streamlit run dashboard/app_streamlit.py
"""

import sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

import theme
import database as db

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "raw"

st.set_page_config(page_title="Centre de Décision — Risque & Marchés", layout="wide",
                    initial_sidebar_state="expanded")

db.initialize_platform_db()

# ---------------------------------------------------------------------------
# Thème (bascule clair / sombre)
# ---------------------------------------------------------------------------
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "dark"

theme.inject_css(st.session_state.theme_mode)
P = theme.PALETTE[st.session_state.theme_mode]

MODEL_STATUS = {
    "Octroi de Crédit": MODEL_DIR / "credit_scoring_model.joblib",
    "Surveillance des Transactions": MODEL_DIR / "fraud_xgboost.joblib",
    "Segmentation Clientèle": MODEL_DIR / "clustering_model.joblib",
    "Veille des Marchés": MODEL_DIR / "nlp_sentiment_baseline.joblib",
    "Prévision Financière": MODEL_DIR / "lstm_stock_forecaster.pt",
}

# ---------------------------------------------------------------------------
# Barre latérale : navigation unique + bascule de thème
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"""
    <div style="padding: 0.4rem 0.2rem 1.1rem 0.2rem;">
        <div style="font-size:1.05rem; font-weight:800; color:{P['text_primary']};">
            Centre de Décision
        </div>
        <div style="font-size:0.75rem; color:{P['text_secondary']};">
            Risque Crédit · Fraude · Marchés
        </div>
    </div>
    """, unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=["Accueil", "Octroi de Crédit", "Surveillance des Transactions",
                 "Segmentation Clientèle", "Veille des Marchés", "Prévision Financière"],
        icons=["speedometer2", "credit-card", "shield-check", "people", "newspaper", "graph-up-arrow"],
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "transparent"},
            "icon": {"color": P["accent_light"], "font-size": "15px"},
            "nav-link": {
                "font-size": "0.86rem", "font-weight": "500", "color": P["text_secondary"],
                "border-radius": "9px", "margin": "3px 0", "padding": "0.55rem 0.8rem",
            },
            "nav-link-selected": {
                "background-color": P["accent_soft"], "color": P["text_primary"], "font-weight": "600",
            },
        },
    )

    st.markdown("<div style='margin-top:1.4rem;'></div>", unsafe_allow_html=True)
    st.caption("Apparence")
    mode_choice = st.radio("Apparence", ["Sombre", "Claire"], horizontal=True, label_visibility="collapsed",
                            index=0 if st.session_state.theme_mode == "dark" else 1)
    new_mode = "dark" if mode_choice == "Sombre" else "light"
    if new_mode != st.session_state.theme_mode:
        st.session_state.theme_mode = new_mode
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("État des services")
    for name, path in MODEL_STATUS.items():
        level = "ok" if path.exists() else "critical"
        label = "Opérationnel" if path.exists() else "Indisponible"
        st.markdown(theme.status_pill(f"{name} — {label}", level), unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom:6px;'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE : ACCUEIL — Tableau de bord de décision
# ---------------------------------------------------------------------------
def render_home():
    theme.page_header(
        "Vue d'ensemble",
        f"Dernière actualisation — {datetime.now().strftime('%d %B %Y, %H:%M')}",
    )

    kpis = db.get_kpis()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(theme.kpi_card(
            "Dossiers de crédit en attente", f"{kpis['pending_credit_applications']}",
            "à instruire"), unsafe_allow_html=True)
    with c2:
        st.markdown(theme.kpi_card(
            "Alertes critiques", f"{kpis['critical_fraud_alerts']}",
            "transactions suspectes non vérifiées"), unsafe_allow_html=True)
    with c3:
        sentiment_label = {"positive": "Favorable", "negative": "Défavorable", "neutral": "Neutre"}.get(
            kpis["dominant_sentiment"], "—")
        st.markdown(theme.kpi_card(
            "Climat de marché", sentiment_label,
            "tendance dominante des actualités"), unsafe_allow_html=True)
    with c4:
        trend = kpis["last_stock_trend_pct"]
        trend_txt = f"{trend:+.2f}%" if trend is not None else "—"
        st.markdown(theme.kpi_card(
            "Tendance boursière", trend_txt,
            "dernière prévision générée"), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.6rem;'></div>", unsafe_allow_html=True)

    left, right = st.columns([2.1, 1])

    with left:
        st.markdown('<div class="section-eyebrow">Modules</div>', unsafe_allow_html=True)
        modules = [
            ("Octroi de Crédit", f"{kpis['avg_default_probability']*100:.1f}%",
             "probabilité moyenne de défaut sur les dossiers analysés"),
            ("Surveillance des Transactions", f"{kpis['total_fraud_checked']}",
             "transactions passées au crible depuis 72h"),
            ("Segmentation Clientèle", f"{kpis['total_customers_segmented']:,}".replace(",", " "),
             "clients répartis en 4 profils comportementaux"),
            ("Veille des Marchés", sentiment_label,
             "sentiment dominant des actualités financières"),
        ]
        mcols = st.columns(2)
        for i, (title, metric, caption) in enumerate(modules):
            status_html = theme.status_pill("Opérationnel", "ok")
            with mcols[i % 2]:
                st.markdown(theme.module_card(title, metric, caption, status_html), unsafe_allow_html=True)
                st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-eyebrow">Notifications</div>', unsafe_allow_html=True)
        notifs = db.get_notifications(limit=6)
        if notifs.empty:
            st.markdown('<div class="section-panel">Aucun événement notable pour le moment.</div>',
                        unsafe_allow_html=True)
        else:
            html = ""
            for _, row in notifs.iterrows():
                try:
                    t = datetime.fromisoformat(row["created_at"]).strftime("%d/%m à %Hh%M")
                except Exception:
                    t = ""
                html += theme.notification_item(row["message"], t, row["severity"])
            st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE : OCTROI DE CRÉDIT
# ---------------------------------------------------------------------------
def render_credit_page():
    theme.page_header("Octroi de Crédit", "Évaluation du risque de défaut et instruction des dossiers",
                       theme.status_pill("Opérationnel", "ok"))

    df_hist = db.read_table("credit_applications", limit=1000)

    tab_summary, tab_explore, tab_action = st.tabs(["Sommaire", "Exploration", "Action"])

    with tab_summary:
        c1, c2, c3 = st.columns(3)
        avg_p = df_hist["default_probability"].mean() * 100 if not df_hist.empty else 0
        n_high = (df_hist["risk_level"] == "high").sum() if not df_hist.empty else 0
        n_pending = (df_hist["status"] == "en_attente").sum() if not df_hist.empty else 0
        c1.markdown(theme.kpi_card("Probabilité moyenne de défaut", f"{avg_p:.1f}%", "sur 1000 derniers dossiers"),
                    unsafe_allow_html=True)
        c2.markdown(theme.kpi_card("Dossiers à risque élevé", f"{n_high}", "nécessitent une analyse manuelle"),
                    unsafe_allow_html=True)
        c3.markdown(theme.kpi_card("En attente d'instruction", f"{n_pending}", "à traiter"),
                    unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1.3rem;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-eyebrow">Nouvelle évaluation</div>', unsafe_allow_html=True)

        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input("Âge", 18, 100, 35)
                income = st.number_input("Revenu mensuel (MAD)", 0, 100000, 4500)
                debt_ratio = st.slider("Taux d'endettement", 0.0, 1.0, 0.3)
            with col2:
                num_credit_lines = st.number_input("Nombre de lignes de crédit", 0, 40, 5)
                num_dependents = st.number_input("Personnes à charge", 0, 10, 1)
                revolving_util = st.slider("Taux d'utilisation renouvelable", 0.0, 1.5, 0.4)
            with col3:
                late_30_59 = st.number_input("Retards 30-59 jours", 0, 20, 0)
                late_60_89 = st.number_input("Retards 60-89 jours", 0, 20, 0)
                late_90 = st.number_input("Retards 90 jours et plus", 0, 20, 0)

            if st.button("Évaluer le dossier", type="primary"):
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
                    risk_level = "low" if proba < 0.3 else "medium" if proba < 0.6 else "high"
                    db.log_credit_application(client, proba, risk_level)

                    risk_pct = proba * 100
                    gc1, gc2 = st.columns([1, 1.4])
                    with gc1:
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number", value=risk_pct,
                            title={"text": "Probabilité de défaut (%)"},
                            gauge={"axis": {"range": [0, 100]},
                                   "bar": {"color": P["critical"] if risk_pct > 50 else P["success"]},
                                   "bgcolor": P["panel"],
                                   "steps": [{"range": [0, 30], "color": P["accent_soft"]},
                                             {"range": [30, 60], "color": P["warning"]},
                                             {"range": [60, 100], "color": P["critical"]}]},
                        ))
                        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color=P["text_primary"], height=260)
                        st.plotly_chart(fig, use_container_width=True)
                    with gc2:
                        if risk_pct < 30:
                            st.markdown(theme.status_pill("Risque faible — crédit recommandé", "ok"),
                                        unsafe_allow_html=True)
                        elif risk_pct < 60:
                            st.markdown(theme.status_pill("Risque modéré — analyse complémentaire", "warning"),
                                        unsafe_allow_html=True)
                        else:
                            st.markdown(theme.status_pill("Risque élevé — crédit déconseillé", "critical"),
                                        unsafe_allow_html=True)
                        st.write("")
                        st.caption("Ce dossier a été enregistré dans l'historique des instructions.")
                except FileNotFoundError:
                    st.error("Modèle non entraîné. Lancez : `python src/credit_scoring.py`")

    with tab_explore:
        if df_hist.empty:
            st.info("Aucune donnée disponible pour l'exploration.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(df_hist, x="default_probability", nbins=25,
                                    title="Distribution des probabilités de défaut",
                                    color_discrete_sequence=[P["accent"]])
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   font_color=P["text_secondary"])
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                risk_counts = df_hist["risk_level"].value_counts().rename(
                    index={"low": "Faible", "medium": "Modéré", "high": "Élevé"})
                fig2 = px.pie(values=risk_counts.values, names=risk_counts.index,
                              title="Répartition par niveau de risque",
                              color_discrete_sequence=[P["success"], P["warning"], P["critical"]])
                fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color=P["text_secondary"])
                st.plotly_chart(fig2, use_container_width=True)

            fig3 = px.scatter(df_hist, x="monthly_income", y="debt_ratio", color="risk_level",
                               title="Revenu vs Taux d'endettement",
                               color_discrete_map={"low": P["success"], "medium": P["warning"], "high": P["critical"]})
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font_color=P["text_secondary"])
            st.plotly_chart(fig3, use_container_width=True)

    with tab_action:
        if df_hist.empty:
            st.info("Aucun dossier à traiter.")
        else:
            st.markdown('<div class="section-eyebrow">Dossiers en attente d\'instruction</div>',
                        unsafe_allow_html=True)
            pending = df_hist[df_hist["status"] == "en_attente"].copy()
            display_cols = ["id", "created_at", "age", "monthly_income", "debt_ratio",
                             "default_probability", "risk_level"]
            st.dataframe(pending[display_cols].rename(columns={
                "id": "N° dossier", "created_at": "Date", "age": "Âge",
                "monthly_income": "Revenu", "debt_ratio": "Endettement",
                "default_probability": "Prob. défaut", "risk_level": "Risque",
            }), use_container_width=True, height=320)

            csv = pending[display_cols].to_csv(index=False).encode("utf-8")
            st.download_button("Exporter en CSV", csv, "dossiers_credit_en_attente.csv", "text/csv")


# ---------------------------------------------------------------------------
# PAGE : SURVEILLANCE DES TRANSACTIONS
# ---------------------------------------------------------------------------
def render_fraud_page():
    theme.page_header("Surveillance des Transactions", "Détection des transactions frauduleuses en temps réel",
                       theme.status_pill("Opérationnel", "ok"))

    df_hist = db.read_table("fraud_alerts", limit=1000)

    tab_summary, tab_explore, tab_action = st.tabs(["Sommaire", "Exploration", "Action"])

    with tab_summary:
        c1, c2, c3 = st.columns(3)
        n_total = len(df_hist)
        n_suspicious = int(df_hist["is_suspicious"].sum()) if not df_hist.empty else 0
        n_critical_new = int(((df_hist["is_suspicious"] == 1) & (df_hist["status"] == "nouvelle")).sum()) if not df_hist.empty else 0
        c1.markdown(theme.kpi_card("Transactions analysées", f"{n_total}", "dernières 72h"), unsafe_allow_html=True)
        c2.markdown(theme.kpi_card("Transactions suspectes", f"{n_suspicious}", "signalées par le modèle"),
                    unsafe_allow_html=True)
        c3.markdown(theme.kpi_card("Alertes critiques non vérifiées", f"{n_critical_new}", "action requise"),
                    unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1.3rem;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-eyebrow">Analyser une transaction</div>', unsafe_allow_html=True)

        amount = st.number_input("Montant de la transaction (MAD)", 0.0, 50000.0, 120.0)
        hour = st.slider("Heure de la transaction", 0, 23, 14)

        if st.button("Analyser la transaction", type="primary"):
            try:
                from feature_engineering import engineer_fraud_features
                model = joblib.load(MODEL_DIR / "fraud_xgboost.joblib")
                rng = np.random.default_rng(42)
                v_features = {f"V{i}": rng.normal(0, 1) for i in range(1, 29)}
                transaction = {**v_features, "Amount": amount, "Time": hour * 3600}
                df_tx = engineer_fraud_features(pd.DataFrame([transaction]))
                df_tx = df_tx[model.get_booster().feature_names]
                proba = float(model.predict_proba(df_tx)[:, 1][0])
                is_suspicious = proba > 0.5
                db.log_fraud_alert(amount, hour, proba, is_suspicious)

                gc1, gc2 = st.columns([1, 2])
                gc1.metric("Probabilité de fraude", f"{proba*100:.2f}%")
                with gc2:
                    if is_suspicious:
                        st.markdown(theme.status_pill("Transaction suspecte — vérification requise", "critical"),
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(theme.status_pill("Transaction légitime", "ok"), unsafe_allow_html=True)
            except FileNotFoundError:
                st.error("Modèle non entraîné. Lancez : `python src/fraud_detection.py`")

    with tab_explore:
        if df_hist.empty:
            st.info("Aucune donnée disponible.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(df_hist, x="hour_of_day", color="is_suspicious",
                                    title="Transactions par heure de la journée", nbins=24,
                                    color_discrete_map={0: P["accent"], 1: P["critical"]})
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   font_color=P["text_secondary"])
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = px.box(df_hist, x="is_suspicious", y="amount",
                               title="Montant par statut de suspicion",
                               color="is_suspicious",
                               color_discrete_map={0: P["accent"], 1: P["critical"]})
                fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    font_color=P["text_secondary"])
                st.plotly_chart(fig2, use_container_width=True)

    with tab_action:
        if df_hist.empty:
            st.info("Aucune alerte à traiter.")
        else:
            st.markdown('<div class="section-eyebrow">Alertes critiques à vérifier</div>', unsafe_allow_html=True)
            critical = df_hist[(df_hist["is_suspicious"] == 1) & (df_hist["status"] == "nouvelle")].copy()
            display_cols = ["id", "created_at", "amount", "hour_of_day", "fraud_probability"]
            st.dataframe(critical[display_cols].rename(columns={
                "id": "N° alerte", "created_at": "Date", "amount": "Montant",
                "hour_of_day": "Heure", "fraud_probability": "Prob. fraude",
            }), use_container_width=True, height=300)

            csv = critical[display_cols].to_csv(index=False).encode("utf-8")
            st.download_button("Exporter en CSV", csv, "alertes_fraude_critiques.csv", "text/csv")


# ---------------------------------------------------------------------------
# PAGE : SEGMENTATION CLIENTÈLE (avec carte)
# ---------------------------------------------------------------------------
def render_segmentation_page():
    theme.page_header("Segmentation Clientèle", "Cartographie et profils comportementaux de la clientèle",
                       theme.status_pill("Opérationnel", "ok"))

    df_seg = db.read_table("customer_segments", limit=6000)

    tab_summary, tab_map, tab_action = st.tabs(["Sommaire", "Carte", "Action"])

    with tab_summary:
        if df_seg.empty:
            st.info("Segmentation non calculée. Lancez : `python src/clustering.py`")
        else:
            counts = df_seg["segment_name"].value_counts()
            cols = st.columns(len(counts))
            for col, (seg, n) in zip(cols, counts.items()):
                col.markdown(theme.kpi_card(seg, f"{n:,}".replace(",", " "),
                                             f"{n/len(df_seg)*100:.1f}% de la base"), unsafe_allow_html=True)

            st.markdown("<div style='margin-top:1.3rem;'></div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                fig = px.pie(df_seg, names="segment_name", title="Répartition des segments",
                             color_discrete_sequence=[P["accent"], P["accent_light"], P["warning"], P["critical"]])
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color=P["text_secondary"])
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = px.scatter(df_seg, x="annual_income", y="balance", color="segment_name",
                                   size="spending_score", title="Revenu vs Solde bancaire",
                                   opacity=0.6)
                fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    font_color=P["text_secondary"])
                st.plotly_chart(fig2, use_container_width=True)

    with tab_map:
        if df_seg.empty:
            st.info("Segmentation non calculée.")
        else:
            st.markdown('<div class="section-eyebrow">Répartition géographique de la clientèle</div>',
                        unsafe_allow_html=True)
            segment_filter = st.multiselect("Filtrer par segment", options=sorted(df_seg["segment_name"].unique()),
                                             default=sorted(df_seg["segment_name"].unique()))
            df_map = df_seg[df_seg["segment_name"].isin(segment_filter)]

            df_map = df_map.copy()
            df_map["marker_size"] = df_map["balance"].clip(lower=0) + 500  # taille toujours positive pour la carte

            map_style = "carto-darkmatter" if st.session_state.theme_mode == "dark" else "carto-positron"
            fig_map = px.scatter_map(
                df_map, lat="latitude", lon="longitude", color="segment_name",
                size="marker_size", size_max=16, zoom=4.6, height=560,
                hover_name="customer_id",
                hover_data={"city": True, "annual_income": True, "spending_score": True,
                            "balance": True, "marker_size": False, "latitude": False, "longitude": False},
                map_style=map_style,
                color_discrete_sequence=[P["accent_light"], P["success"], P["warning"], P["critical"]],
            )
            fig_map.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=0, b=0),
                                   legend=dict(bgcolor="rgba(0,0,0,0)", font_color=P["text_primary"]))
            st.plotly_chart(fig_map, use_container_width=True)

            st.markdown('<div class="section-eyebrow" style="margin-top:1rem;">Répartition par ville</div>',
                        unsafe_allow_html=True)
            city_seg = df_map.groupby(["city", "segment_name"]).size().reset_index(name="clients")
            fig_bar = px.bar(city_seg, x="city", y="clients", color="segment_name", barmode="stack")
            fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   font_color=P["text_secondary"])
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab_action:
        if df_seg.empty:
            st.info("Aucune donnée à exporter.")
        else:
            st.markdown('<div class="section-eyebrow">Export de la base client segmentée</div>',
                        unsafe_allow_html=True)
            st.dataframe(df_seg.head(200), use_container_width=True, height=320)
            csv = df_seg.to_csv(index=False).encode("utf-8")
            st.download_button("Exporter la segmentation en CSV", csv, "segmentation_clientele.csv", "text/csv")


# ---------------------------------------------------------------------------
# PAGE : VEILLE DES MARCHÉS (NLP Sentiment)
# ---------------------------------------------------------------------------
def render_sentiment_page():
    theme.page_header("Veille des Marchés", "Analyse de sentiment des actualités financières",
                       theme.status_pill("Opérationnel", "ok"))

    df_hist = db.read_table("sentiment_log", limit=500)

    tab_summary, tab_explore, tab_action = st.tabs(["Sommaire", "Exploration", "Action"])

    with tab_summary:
        if not df_hist.empty:
            counts = df_hist["sentiment"].value_counts()
            labels_map = {"positive": "Favorable", "negative": "Défavorable", "neutral": "Neutre"}
            c1, c2, c3 = st.columns(3)
            for col, key in zip([c1, c2, c3], ["positive", "negative", "neutral"]):
                n = int(counts.get(key, 0))
                col.markdown(theme.kpi_card(labels_map[key], f"{n}", "articles analysés"), unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1.3rem;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-eyebrow">Analyser une actualité</div>', unsafe_allow_html=True)
        text = st.text_area("Titre ou extrait d'actualité financière",
                             "Company Alpha Corp reports record profits this quarter")

        if st.button("Analyser le sentiment", type="primary"):
            try:
                from nlp_sentiment import predict_sentiment_baseline
                sentiment = predict_sentiment_baseline(text)
                db.log_sentiment(text, sentiment)
                label = {"positive": "Favorable", "negative": "Défavorable", "neutral": "Neutre"}[sentiment]
                level = {"positive": "ok", "negative": "critical", "neutral": "warning"}[sentiment]
                st.markdown(theme.status_pill(f"Sentiment détecté — {label}", level), unsafe_allow_html=True)
            except FileNotFoundError:
                st.error("Modèle non entraîné. Lancez : `python src/nlp_sentiment.py`")

    with tab_explore:
        if df_hist.empty:
            st.info("Aucune donnée disponible.")
        else:
            fig = px.histogram(df_hist, x="sentiment", title="Distribution des sentiments analysés",
                                color="sentiment",
                                color_discrete_map={"positive": P["success"], "negative": P["critical"],
                                                     "neutral": P["warning"]})
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color=P["text_secondary"])
            st.plotly_chart(fig, use_container_width=True)

    with tab_action:
        if df_hist.empty:
            st.info("Aucun historique à exporter.")
        else:
            st.dataframe(df_hist, use_container_width=True, height=320)
            csv = df_hist.to_csv(index=False).encode("utf-8")
            st.download_button("Exporter l'historique en CSV", csv, "historique_sentiment.csv", "text/csv")


# ---------------------------------------------------------------------------
# PAGE : PRÉVISION FINANCIÈRE (LSTM boursier)
# ---------------------------------------------------------------------------
def render_stock_page():
    theme.page_header("Prévision Financière", "Anticipation des cours boursiers et de la volatilité",
                       theme.status_pill("Opérationnel", "ok"))

    stock_path = DATA_DIR / "stock_prices.csv"
    df_forecasts = db.read_table("stock_forecasts", limit=200)

    tab_summary, tab_explore, tab_action = st.tabs(["Sommaire", "Exploration", "Action"])

    with tab_summary:
        if not df_forecasts.empty:
            last = df_forecasts.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.markdown(theme.kpi_card("Dernier cours de référence", f"{last['last_close']:.2f}", "MAD"),
                        unsafe_allow_html=True)
            c2.markdown(theme.kpi_card("Prévision", f"{last['forecast_close']:.2f}", "MAD"),
                        unsafe_allow_html=True)
            trend_sign = "+" if last["trend_pct"] >= 0 else ""
            c3.markdown(theme.kpi_card("Tendance anticipée", f"{trend_sign}{last['trend_pct']:.2f}%",
                                        f"sur {int(last['horizon_days'])} jour(s)"), unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1.3rem;'></div>", unsafe_allow_html=True)
        if stock_path.exists():
            df_stock = pd.read_csv(stock_path, parse_dates=["Date"])
            n_days = st.slider("Horizon de prévision (jours)", 1, 15, 5)
            if st.button("Générer la prévision", type="primary"):
                try:
                    import torch
                    from time_series_lstm import LSTMForecaster, forecast_next_days, WINDOW_SIZE

                    scaler = joblib.load(MODEL_DIR / "lstm_scaler.joblib")
                    model = LSTMForecaster()
                    model.load_state_dict(torch.load(MODEL_DIR / "lstm_stock_forecaster.pt"))

                    last_window = df_stock["Close"].values[-WINDOW_SIZE:]
                    preds = forecast_next_days(model, scaler, last_window, n_days=n_days)
                    db.log_stock_forecast(n_days, float(df_stock["Close"].iloc[-1]), float(preds[-1]))

                    future_dates = pd.bdate_range(
                        start=df_stock["Date"].iloc[-1] + pd.Timedelta(days=1), periods=n_days)
                    forecast_df = pd.DataFrame({"Date": future_dates, "Close": preds})

                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=df_stock["Date"].tail(60), y=df_stock["Close"].tail(60),
                                               name="Historique", mode="lines", line=dict(color=P["accent"])))
                    fig2.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Close"],
                                               name="Prévision", mode="lines+markers",
                                               line=dict(dash="dash", color=P["warning"])))
                    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                        font_color=P["text_secondary"])
                    st.plotly_chart(fig2, use_container_width=True)
                    st.dataframe(forecast_df, use_container_width=True)
                except FileNotFoundError:
                    st.error("Modèle LSTM non entraîné. Lancez : `python src/time_series_lstm.py`")
        else:
            st.info("Données boursières manquantes. Lancez : `python src/data_generation.py`")

    with tab_explore:
        if stock_path.exists():
            df_stock = pd.read_csv(stock_path, parse_dates=["Date"])
            fig = px.line(df_stock, x="Date", y="Close", title="Historique du cours de clôture")
            fig.update_traces(line_color=P["accent_light"])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color=P["text_secondary"])
            st.plotly_chart(fig, use_container_width=True)

            df_stock["daily_return"] = df_stock["Close"].pct_change()
            fig2 = px.histogram(df_stock, x="daily_return", nbins=50, title="Distribution des rendements journaliers")
            fig2.update_traces(marker_color=P["accent"])
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font_color=P["text_secondary"])
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Aucune donnée disponible.")

    with tab_action:
        if df_forecasts.empty:
            st.info("Aucune prévision enregistrée.")
        else:
            st.dataframe(df_forecasts, use_container_width=True, height=320)
            csv = df_forecasts.to_csv(index=False).encode("utf-8")
            st.download_button("Exporter les prévisions en CSV", csv, "previsions_boursieres.csv", "text/csv")


# ---------------------------------------------------------------------------
# Routage
# ---------------------------------------------------------------------------
PAGES = {
    "Accueil": render_home,
    "Octroi de Crédit": render_credit_page,
    "Surveillance des Transactions": render_fraud_page,
    "Segmentation Clientèle": render_segmentation_page,
    "Veille des Marchés": render_sentiment_page,
    "Prévision Financière": render_stock_page,
}

PAGES[selected]()