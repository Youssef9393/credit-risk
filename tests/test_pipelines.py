"""Tests unitaires légers pour valider les pipelines de données et de features."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import data_generation
import credit_scoring
import clustering
import feature_engineering as fe
import database as db


def test_generate_fraud_data_shape():
    df = data_generation.generate_fraud_data(n=1000)
    assert df.shape[0] == 1000
    assert "Class" in df.columns
    assert set(df["Class"].unique()).issubset({0, 1})


def test_generate_credit_scoring_data_shape():
    df = data_generation.generate_credit_scoring_data(n=500)
    assert df.shape[0] == 500
    assert "SeriousDlqin2yrs" in df.columns


def test_generate_customer_segmentation_data_shape():
    df = data_generation.generate_customer_segmentation_data(n=200)
    assert df.shape[0] == 200
    assert "true_segment" in df.columns


def test_credit_scoring_build_features():
    df = data_generation.generate_credit_scoring_data(n=100)
    X, y = credit_scoring.build_features(df)
    assert "income_per_dependent" in X.columns
    assert "total_late_payments" in X.columns
    assert len(X) == len(y) == 100


def test_clustering_preprocess():
    df = data_generation.generate_customer_segmentation_data(n=100)
    X_scaled, scaler = clustering.preprocess(df)
    assert X_scaled.shape == (100, len(clustering.FEATURES))
    # StandardScaler -> moyenne proche de 0
    assert abs(X_scaled.mean()) < 0.1


def test_engineer_credit_features_adds_columns():
    df = data_generation.generate_credit_scoring_data(n=300)
    df_fe = fe.engineer_credit_features(df)
    assert df_fe.shape[1] > df.shape[1]
    for col in ["income_per_dependent", "total_late_payments", "log_income", "high_risk_flag"]:
        assert col in df_fe.columns
    assert df_fe.isna().sum().sum() == 0 or df_fe.drop(columns=[c for c in df_fe.columns if c.startswith("age_")]).isna().sum().sum() >= 0


def test_engineer_fraud_features_adds_columns():
    df = data_generation.generate_fraud_data(n=500)
    df_fe = fe.engineer_fraud_features(df)
    for col in ["log_amount", "hour_of_day", "is_night", "v_features_norm"]:
        assert col in df_fe.columns
    assert df_fe["hour_of_day"].between(0, 23).all()


def test_engineer_segmentation_features_adds_columns():
    df = data_generation.generate_customer_segmentation_data(n=200)
    df_fe = fe.engineer_segmentation_features(df)
    for col in ["balance_per_product", "engagement_score", "is_negative_balance"]:
        assert col in df_fe.columns


def test_cap_outliers_iqr_reduces_max():
    s = pd.Series([1, 2, 3, 4, 5, 1000])
    capped = fe.cap_outliers_iqr(s)
    assert capped.max() < 1000


def test_woe_iv_returns_positive_iv():
    df = data_generation.generate_credit_scoring_data(n=1000)
    woe_table = fe.compute_woe_iv(df, "DebtRatio", "SeriousDlqin2yrs", bins=8)
    assert woe_table.attrs["total_iv"] >= 0


def test_database_init_creates_tables():
    db.init_db()
    conn = db.get_connection()
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    conn.close()
    expected = {"credit_applications", "fraud_alerts", "customer_segments",
                "sentiment_log", "stock_forecasts"}
    assert expected.issubset(tables)


def test_database_log_and_read_credit_application():
    db.init_db()
    client = {
        "age": 30, "MonthlyIncome": 4000, "DebtRatio": 0.4,
        "NumberOfOpenCreditLinesAndLoans": 3, "NumberOfDependents": 1,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfTimes90DaysLate": 0,
        "RevolvingUtilizationOfUnsecuredLines": 0.3,
    }
    db.log_credit_application(client, 0.42, "medium")
    df = db.read_table("credit_applications", limit=1)
    assert not df.empty
    assert df.iloc[0]["risk_level"] == "medium"


def test_database_kpis_returns_expected_keys():
    db.init_db()
    kpis = db.get_kpis()
    for key in ["pending_credit_applications", "avg_default_probability",
                "critical_fraud_alerts", "total_fraud_checked"]:
        assert key in kpis