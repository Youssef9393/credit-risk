"""
Module Base de données (SQLite)
====================================
Persiste toutes les données opérationnelles de la plateforme :
 - Dossiers de crédit analysés (credit_applications)
 - Alertes de fraude (fraud_alerts)
 - Segments clients géolocalisés (customer_segments)
 - Historique d'analyse de sentiment (sentiment_log)
 - Prévisions boursières (stock_forecasts)

SQLite est utilisé pour la simplicité de déploiement (fichier unique,
zéro configuration). Pour un environnement de production multi-utilisateurs,
remplacez la chaîne de connexion par PostgreSQL/MySQL sans changer l'API
de ce module (les fonctions restent identiques).
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "platform.db"

# Villes marocaines utilisées pour géolocaliser les clients synthétiques sur la carte
CITIES = {
    "Casablanca": (33.5731, -7.5898),
    "Rabat": (34.0209, -6.8416),
    "Fès": (34.0331, -5.0003),
    "Marrakech": (31.6295, -7.9811),
    "Tanger": (35.7595, -5.8340),
    "Agadir": (30.4278, -9.5981),
    "Meknès": (33.8935, -5.5473),
    "Oujda": (34.6805, -1.9086),
    "Kénitra": (34.2610, -6.5802),
    "Tétouan": (35.5785, -5.3684),
}


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS credit_applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            age INTEGER, monthly_income REAL, debt_ratio REAL,
            num_credit_lines INTEGER, num_dependents INTEGER,
            late_30_59 INTEGER, late_60_89 INTEGER, late_90 INTEGER,
            revolving_util REAL,
            default_probability REAL, risk_level TEXT,
            status TEXT DEFAULT 'en_attente'
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS fraud_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            amount REAL, hour_of_day INTEGER,
            fraud_probability REAL, is_suspicious INTEGER,
            status TEXT DEFAULT 'nouvelle'
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS customer_segments (
            customer_id TEXT PRIMARY KEY,
            age INTEGER, annual_income REAL, balance REAL,
            spending_score REAL, tenure_years REAL, num_products INTEGER,
            segment_name TEXT, city TEXT, latitude REAL, longitude REAL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            text TEXT, sentiment TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS stock_forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            horizon_days INTEGER, last_close REAL,
            forecast_close REAL, trend_pct REAL
        )
    """)

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Insertion (appelée par le dashboard à chaque nouvelle prédiction utilisateur)
# ---------------------------------------------------------------------------
def log_credit_application(payload: dict, proba: float, risk_level: str):
    conn = get_connection()
    conn.execute("""
        INSERT INTO credit_applications
        (created_at, age, monthly_income, debt_ratio, num_credit_lines, num_dependents,
         late_30_59, late_60_89, late_90, revolving_util, default_probability, risk_level)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.now().isoformat(), payload["age"], payload["MonthlyIncome"], payload["DebtRatio"],
        payload["NumberOfOpenCreditLinesAndLoans"], payload["NumberOfDependents"],
        payload["NumberOfTime30-59DaysPastDueNotWorse"], payload["NumberOfTime60-89DaysPastDueNotWorse"],
        payload["NumberOfTimes90DaysLate"], payload["RevolvingUtilizationOfUnsecuredLines"],
        proba, risk_level,
    ))
    conn.commit()
    conn.close()


def log_fraud_alert(amount: float, hour: int, proba: float, is_suspicious: bool):
    conn = get_connection()
    conn.execute("""
        INSERT INTO fraud_alerts (created_at, amount, hour_of_day, fraud_probability, is_suspicious)
        VALUES (?,?,?,?,?)
    """, (datetime.now().isoformat(), amount, hour, proba, int(is_suspicious)))
    conn.commit()
    conn.close()


def log_sentiment(text: str, sentiment: str):
    conn = get_connection()
    conn.execute("""
        INSERT INTO sentiment_log (created_at, text, sentiment) VALUES (?,?,?)
    """, (datetime.now().isoformat(), text, sentiment))
    conn.commit()
    conn.close()


def log_stock_forecast(horizon_days: int, last_close: float, forecast_close: float):
    trend_pct = (forecast_close - last_close) / last_close * 100
    conn = get_connection()
    conn.execute("""
        INSERT INTO stock_forecasts (created_at, horizon_days, last_close, forecast_close, trend_pct)
        VALUES (?,?,?,?,?)
    """, (datetime.now().isoformat(), horizon_days, last_close, forecast_close, trend_pct))
    conn.commit()
    conn.close()


def update_status(table: str, row_id: int, status: str):
    assert table in {"credit_applications", "fraud_alerts"}
    conn = get_connection()
    conn.execute(f"UPDATE {table} SET status = ? WHERE id = ?", (status, row_id))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Lecture (utilisée pour les KPIs et tableaux du dashboard)
# ---------------------------------------------------------------------------
def read_table(table: str, limit: int = 500) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query(f"SELECT * FROM {table} ORDER BY ROWID DESC LIMIT {limit}", conn)
    conn.close()
    return df


def get_kpis() -> dict:
    conn = get_connection()
    kpis = {}

    kpis["pending_credit_applications"] = conn.execute(
        "SELECT COUNT(*) FROM credit_applications WHERE status = 'en_attente'"
    ).fetchone()[0]

    kpis["avg_default_probability"] = conn.execute(
        "SELECT AVG(default_probability) FROM credit_applications"
    ).fetchone()[0] or 0.0

    kpis["critical_fraud_alerts"] = conn.execute(
        "SELECT COUNT(*) FROM fraud_alerts WHERE is_suspicious = 1 AND status = 'nouvelle'"
    ).fetchone()[0]

    kpis["total_fraud_checked"] = conn.execute("SELECT COUNT(*) FROM fraud_alerts").fetchone()[0]

    row = conn.execute(
        "SELECT sentiment, COUNT(*) c FROM sentiment_log GROUP BY sentiment ORDER BY c DESC LIMIT 1"
    ).fetchone()
    kpis["dominant_sentiment"] = row["sentiment"] if row else None

    row = conn.execute(
        "SELECT trend_pct FROM stock_forecasts ORDER BY id DESC LIMIT 1"
    ).fetchone()
    kpis["last_stock_trend_pct"] = row["trend_pct"] if row else None

    kpis["total_customers_segmented"] = conn.execute(
        "SELECT COUNT(*) FROM customer_segments"
    ).fetchone()[0]

    conn.close()
    return kpis


def get_notifications(limit: int = 8) -> pd.DataFrame:
    """Regroupe les derniers événements notables (alertes fraude critiques, dossiers à haut risque)."""
    conn = get_connection()
    fraud = pd.read_sql_query(f"""
        SELECT created_at, 'Fraude' as category,
               'Transaction suspecte de ' || ROUND(amount,2) || ' MAD (proba ' || ROUND(fraud_probability*100,1) || '%)' as message,
               'critique' as severity
        FROM fraud_alerts WHERE is_suspicious = 1
        ORDER BY id DESC LIMIT {limit}
    """, conn)

    credit = pd.read_sql_query(f"""
        SELECT created_at, 'Crédit' as category,
               'Dossier à risque élevé (proba défaut ' || ROUND(default_probability*100,1) || '%)' as message,
               'attention' as severity
        FROM credit_applications WHERE risk_level = 'high'
        ORDER BY id DESC LIMIT {limit}
    """, conn)
    conn.close()

    notifications = pd.concat([fraud, credit], ignore_index=True)
    if not notifications.empty:
        notifications = notifications.sort_values("created_at", ascending=False).head(limit)
    return notifications


# ---------------------------------------------------------------------------
# Seed : peuple la base avec un historique réaliste + segmentation géolocalisée
# ---------------------------------------------------------------------------
def seed_customer_segments(force=False):
    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM customer_segments").fetchone()[0]
    if count > 0 and not force:
        conn.close()
        return

    seg_path = BASE_DIR / "data" / "customer_segments.csv"
    if not seg_path.exists():
        conn.close()
        return

    df = pd.read_csv(seg_path)
    rng = np.random.default_rng(42)
    city_names = list(CITIES.keys())
    df["city"] = rng.choice(city_names, size=len(df))
    df["latitude"] = df["city"].map(lambda c: CITIES[c][0]) + rng.normal(0, 0.06, size=len(df))
    df["longitude"] = df["city"].map(lambda c: CITIES[c][1]) + rng.normal(0, 0.06, size=len(df))

    conn.execute("DELETE FROM customer_segments")
    df[["customer_id", "age", "annual_income", "balance", "spending_score", "tenure_years",
        "num_products", "segment_name", "city", "latitude", "longitude"]].to_sql(
        "customer_segments", conn, if_exists="append", index=False
    )
    conn.commit()
    conn.close()


def seed_history(n_credit=180, n_fraud=220, n_sentiment=40, seed=7, force=False):
    """Génère un historique réaliste des derniers jours pour que le dashboard
    affiche des KPIs pertinents dès le premier lancement."""
    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM credit_applications").fetchone()[0]
    if count > 0 and not force:
        conn.close()
        return

    rng = np.random.default_rng(seed)
    now = datetime.now()

    conn.execute("DELETE FROM credit_applications")
    conn.execute("DELETE FROM fraud_alerts")
    conn.execute("DELETE FROM sentiment_log")
    conn.execute("DELETE FROM stock_forecasts")

    for _ in range(n_credit):
        ts = now - timedelta(hours=float(rng.uniform(0, 72)))
        proba = float(np.clip(rng.beta(2, 5), 0, 1))
        risk = "low" if proba < 0.3 else "medium" if proba < 0.6 else "high"
        status = rng.choice(["en_attente", "traite"], p=[0.35, 0.65])
        conn.execute("""
            INSERT INTO credit_applications
            (created_at, age, monthly_income, debt_ratio, num_credit_lines, num_dependents,
             late_30_59, late_60_89, late_90, revolving_util, default_probability, risk_level, status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            ts.isoformat(), int(rng.integers(21, 75)), float(rng.uniform(2000, 15000)),
            float(rng.uniform(0.05, 0.8)), int(rng.integers(1, 15)), int(rng.integers(0, 4)),
            int(rng.poisson(0.3)), int(rng.poisson(0.15)), int(rng.poisson(0.1)),
            float(rng.uniform(0.05, 1.0)), proba, risk, status,
        ))

    for _ in range(n_fraud):
        ts = now - timedelta(hours=float(rng.uniform(0, 72)))
        proba = float(np.clip(rng.beta(1.2, 20), 0, 1))
        suspicious = proba > 0.5
        status = "nouvelle" if suspicious and rng.random() < 0.4 else "verifiee"
        conn.execute("""
            INSERT INTO fraud_alerts (created_at, amount, hour_of_day, fraud_probability, is_suspicious, status)
            VALUES (?,?,?,?,?,?)
        """, (ts.isoformat(), float(rng.exponential(300)), int(rng.integers(0, 24)),
              proba, int(suspicious), status))

    # Garantit quelques alertes critiques récentes non traitées, pour une démo KPI parlante
    for _ in range(4):
        ts = now - timedelta(hours=float(rng.uniform(0, 6)))
        proba = float(rng.uniform(0.7, 0.97))
        conn.execute("""
            INSERT INTO fraud_alerts (created_at, amount, hour_of_day, fraud_probability, is_suspicious, status)
            VALUES (?,?,?,?,?,?)
        """, (ts.isoformat(), float(rng.uniform(500, 4000)), int(rng.integers(0, 24)),
              proba, 1, "nouvelle"))

    for _ in range(3):
        ts = now - timedelta(hours=float(rng.uniform(0, 6)))
        proba = float(rng.uniform(0.65, 0.9))
        conn.execute("""
            INSERT INTO credit_applications
            (created_at, age, monthly_income, debt_ratio, num_credit_lines, num_dependents,
             late_30_59, late_60_89, late_90, revolving_util, default_probability, risk_level, status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            ts.isoformat(), int(rng.integers(21, 75)), float(rng.uniform(2000, 15000)),
            float(rng.uniform(0.5, 0.9)), int(rng.integers(1, 15)), int(rng.integers(0, 4)),
            int(rng.integers(1, 5)), int(rng.integers(0, 3)), int(rng.integers(0, 2)),
            float(rng.uniform(0.7, 1.0)), proba, "high", "en_attente",
        ))

    sentiments = rng.choice(["positive", "negative", "neutral"], size=n_sentiment, p=[0.45, 0.25, 0.30])
    for s in sentiments:
        ts = now - timedelta(hours=float(rng.uniform(0, 72)))
        conn.execute("INSERT INTO sentiment_log (created_at, text, sentiment) VALUES (?,?,?)",
                      (ts.isoformat(), "(historique)", s))

    last_close = 118.4
    for i in range(10):
        ts = now - timedelta(hours=float(rng.uniform(0, 72)))
        trend = float(rng.normal(0.3, 1.2))
        forecast = last_close * (1 + trend / 100)
        conn.execute("""
            INSERT INTO stock_forecasts (created_at, horizon_days, last_close, forecast_close, trend_pct)
            VALUES (?,?,?,?,?)
        """, (ts.isoformat(), int(rng.integers(1, 10)), last_close, forecast, trend))

    conn.commit()
    conn.close()


def initialize_platform_db():
    """Point d'entrée unique : crée les tables et peuple les données de démo si besoin."""
    init_db()
    seed_history()
    seed_customer_segments()


if __name__ == "__main__":
    initialize_platform_db()
    print(f"Base de données initialisée -> {DB_PATH}")
    print(get_kpis())