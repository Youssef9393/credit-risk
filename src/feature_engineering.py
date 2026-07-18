"""
Module de Feature Engineering
=================================
Fonctions réutilisables de préparation et création de variables pour :
 - Credit Scoring
 - Détection de fraude
 - Segmentation client

Contient : gestion des valeurs manquantes/aberrantes, transformations,
variables d'interaction, binning, Weight of Evidence (WOE) / Information Value (IV),
et sélection de variables (corrélation, importance).

Usage:
    from feature_engineering import engineer_credit_features, engineer_fraud_features
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Utilitaires génériques
# ---------------------------------------------------------------------------
def cap_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """Winsorisation par IQR : plafonne les valeurs extrêmes plutôt que de les supprimer."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return series.clip(lower, upper)


def log1p_transform(series: pd.Series) -> pd.Series:
    """Transformation log(1+x), utile pour les variables très asymétriques (revenu, montant...)."""
    return np.log1p(series.clip(lower=0))


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Résumé du taux de valeurs manquantes par colonne."""
    report = pd.DataFrame({
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(2),
        "dtype": df.dtypes.astype(str),
    })
    return report.sort_values("pct_missing", ascending=False)


def compute_woe_iv(df: pd.DataFrame, feature: str, target: str, bins: int = 10) -> pd.DataFrame:
    """
    Calcule le Weight of Evidence (WOE) et l'Information Value (IV) d'une variable
    par rapport à une cible binaire — standard en credit scoring pour évaluer
    le pouvoir prédictif d'une variable avant modélisation.
    IV < 0.02 : non prédictif | 0.02-0.1 : faible | 0.1-0.3 : moyen | >0.3 : fort
    """
    tmp = df[[feature, target]].copy()
    try:
        tmp["bin"] = pd.qcut(tmp[feature], q=bins, duplicates="drop")
    except ValueError:
        tmp["bin"] = pd.cut(tmp[feature], bins=bins)

    grouped = tmp.groupby("bin", observed=True)[target].agg(["count", "sum"])
    grouped.columns = ["total", "bad"]
    grouped["good"] = grouped["total"] - grouped["bad"]

    total_bad = grouped["bad"].sum()
    total_good = grouped["good"].sum()

    grouped["pct_bad"] = (grouped["bad"] / total_bad).replace(0, 1e-6)
    grouped["pct_good"] = (grouped["good"] / total_good).replace(0, 1e-6)
    grouped["woe"] = np.log(grouped["pct_good"] / grouped["pct_bad"])
    grouped["iv"] = (grouped["pct_good"] - grouped["pct_bad"]) * grouped["woe"]

    grouped.attrs["total_iv"] = grouped["iv"].sum()
    return grouped


def feature_importance_report(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Combine importance Random Forest + information mutuelle pour classer les variables."""
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    mi = mutual_info_classif(X, y, random_state=42)

    report = pd.DataFrame({
        "feature": X.columns,
        "rf_importance": rf.feature_importances_,
        "mutual_info": mi,
    }).sort_values("rf_importance", ascending=False).reset_index(drop=True)
    return report


def correlation_filter(X: pd.DataFrame, threshold: float = 0.9) -> list:
    """Retourne la liste des colonnes à supprimer car trop corrélées (> threshold) avec une autre."""
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return to_drop


# ---------------------------------------------------------------------------
# Feature engineering spécifique : Credit Scoring
# ---------------------------------------------------------------------------
def engineer_credit_features(df: pd.DataFrame, target: str = "SeriousDlqin2yrs") -> pd.DataFrame:
    """
    Pipeline de feature engineering pour le credit scoring.
    Crée des ratios financiers, transforme les variables asymétriques,
    plafonne les outliers, et découpe l'âge/revenu en tranches (binning).
    """
    df = df.copy()

    # 1. Traitement des valeurs manquantes (médiane, robuste aux outliers)
    for col in ["MonthlyIncome", "NumberOfDependents"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # 2. Plafonnement des outliers (IQR) sur les variables sensibles
    for col in ["DebtRatio", "RevolvingUtilizationOfUnsecuredLines", "MonthlyIncome"]:
        if col in df.columns:
            df[col] = cap_outliers_iqr(df[col])

    # 3. Transformation log des variables asymétriques
    df["log_income"] = log1p_transform(df["MonthlyIncome"])

    # 4. Variables d'interaction / ratios financiers
    df["income_per_dependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)
    df["total_late_payments"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"]
        + df["NumberOfTime60-89DaysPastDueNotWorse"]
        + df["NumberOfTimes90DaysLate"]
    )
    df["has_been_late"] = (df["total_late_payments"] > 0).astype(int)
    df["credit_lines_per_dependent"] = df["NumberOfOpenCreditLinesAndLoans"] / (df["NumberOfDependents"] + 1)
    df["debt_to_income_interaction"] = df["DebtRatio"] * df["log_income"]
    df["utilization_x_late"] = df["RevolvingUtilizationOfUnsecuredLines"] * (1 + df["total_late_payments"])

    # 5. Binning de l'âge (tranches métier)
    df["age_bin"] = pd.cut(
        df["age"], bins=[0, 25, 35, 45, 55, 65, 120],
        labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"],
    )
    df = pd.get_dummies(df, columns=["age_bin"], prefix="age", drop_first=True)

    # 6. Indicateur de "client à haut risque structurel"
    df["high_risk_flag"] = (
        (df["RevolvingUtilizationOfUnsecuredLines"] > 0.8) & (df["total_late_payments"] > 2)
    ).astype(int)

    return df


# ---------------------------------------------------------------------------
# Feature engineering spécifique : Détection de fraude
# ---------------------------------------------------------------------------
def engineer_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline de feature engineering pour la détection de fraude.
    Transforme le montant, extrait des features temporelles,
    et crée des indicateurs de comportement inhabituel.
    """
    df = df.copy()

    # 1. Transformation log du montant (très asymétrique)
    df["log_amount"] = log1p_transform(df["Amount"])

    # 2. Features temporelles à partir de "Time" (secondes écoulées depuis le début)
    df["hour_of_day"] = ((df["Time"] % 86400) // 3600).astype(int)
    df["is_night"] = df["hour_of_day"].apply(lambda h: 1 if (h < 6 or h >= 22) else 0)
    df["day_index"] = (df["Time"] // 86400).astype(int)

    # 3. Montant relatif : écart au montant moyen (z-score simplifié)
    amount_mean, amount_std = df["Amount"].mean(), df["Amount"].std()
    df["amount_zscore"] = (df["Amount"] - amount_mean) / (amount_std + 1e-6)
    df["is_large_amount"] = (df["Amount"] > df["Amount"].quantile(0.99)).astype(int)

    # 4. Statistique agrégée des variables PCA (V1-V28) : magnitude du vecteur
    v_cols = [c for c in df.columns if c.startswith("V")]
    if v_cols:
        df["v_features_norm"] = np.sqrt((df[v_cols] ** 2).sum(axis=1))
        df["v_features_mean"] = df[v_cols].mean(axis=1)

    return df


# ---------------------------------------------------------------------------
# Feature engineering spécifique : Segmentation client
# ---------------------------------------------------------------------------
def engineer_segmentation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline de feature engineering pour la segmentation client.
    Crée des ratios de type RFM (Recency/Frequency/Monetary) approximés
    à partir des variables disponibles (solde, ancienneté, produits détenus).
    """
    df = df.copy()

    df["balance_per_product"] = df["balance"] / df["num_products"].clip(lower=1)
    df["income_to_age_ratio"] = df["annual_income"] / df["age"].clip(lower=18)
    df["tenure_ratio"] = df["tenure_years"] / df["age"].clip(lower=18)
    df["engagement_score"] = (
        0.4 * df["spending_score"]
        + 0.3 * (df["num_products"] / df["num_products"].max() * 100)
        + 0.3 * (df["tenure_years"] / df["tenure_years"].max() * 100)
    )
    df["is_negative_balance"] = (df["balance"] < 0).astype(int)
    df["log_income"] = log1p_transform(df["annual_income"])

    return df


if __name__ == "__main__":
    import data_generation

    print(">>> Test rapide du module de feature engineering\n")

    df_credit = data_generation.generate_credit_scoring_data(n=2000)
    df_credit_fe = engineer_credit_features(df_credit)
    print(f"Credit scoring : {df_credit.shape[1]} -> {df_credit_fe.shape[1]} colonnes après FE")

    df_fraud = data_generation.generate_fraud_data(n=2000)
    df_fraud_fe = engineer_fraud_features(df_fraud)
    print(f"Fraude : {df_fraud.shape[1]} -> {df_fraud_fe.shape[1]} colonnes après FE")

    df_seg = data_generation.generate_customer_segmentation_data(n=2000)
    df_seg_fe = engineer_segmentation_features(df_seg)
    print(f"Segmentation : {df_seg.shape[1]} -> {df_seg_fe.shape[1]} colonnes après FE")

    print("\n>>> Rapport d'importance des variables (Credit Scoring)")
    X = df_credit_fe.drop(columns=["SeriousDlqin2yrs"]).select_dtypes(include=[np.number])
    y = df_credit_fe["SeriousDlqin2yrs"]
    report = feature_importance_report(X, y)
    print(report.head(10).round(4))
