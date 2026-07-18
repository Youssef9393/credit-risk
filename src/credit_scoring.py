"""
Module Credit Scoring
======================
Prédit la probabilité qu'un client fasse défaut (SeriousDlqin2yrs)
à partir des features financières (revenu, endettement, retards de paiement...).

Usage:
    python src/credit_scoring.py
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)
import xgboost as xgb
import lightgbm as lgb

from feature_engineering import engineer_credit_features

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "credit_scoring.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "SeriousDlqin2yrs"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def build_features(df: pd.DataFrame):
    """Applique le pipeline complet de feature engineering (voir src/feature_engineering.py :
    ratios financiers, binning age, plafonnement outliers, log-transform...) puis sépare X / y."""
    df_fe = engineer_credit_features(df, target=TARGET)
    X = df_fe.drop(columns=[TARGET])
    y = df_fe[TARGET]
    return X, y


def get_models():
    return {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=10, class_weight="balanced",
                n_jobs=-1, random_state=42)),
        ]),
        "xgboost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", xgb.XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="auc", random_state=42)),
        ]),
        "lightgbm": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", lgb.LGBMClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                random_state=42, verbose=-1)),
        ]),
    }


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def train_all(save_best=True):
    df = load_data()
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    results = {}
    models = get_models()
    for name, pipe in models.items():
        print(f"\n--- Entraînement: {name} ---")
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test)
        results[name] = metrics
        print(pd.Series(metrics).round(4))

    results_df = pd.DataFrame(results).T.sort_values("roc_auc", ascending=False)
    print("\n=== Comparatif des modèles ===")
    print(results_df.round(4))

    best_name = results_df.index[0]
    best_model = models[best_name]
    print(f"\nMeilleur modèle : {best_name} (ROC-AUC={results_df.loc[best_name, 'roc_auc']:.4f})")

    if save_best:
        joblib.dump(best_model, MODEL_DIR / "credit_scoring_model.joblib")
        joblib.dump(list(X.columns), MODEL_DIR / "credit_scoring_features.joblib")
        print(f"Modèle sauvegardé -> {MODEL_DIR / 'credit_scoring_model.joblib'}")

    return best_model, results_df


def predict_default_probability(client: dict, model=None) -> float:
    """Prédit la probabilité de défaut pour un client donné (dict de features brutes)."""
    if model is None:
        model = joblib.load(MODEL_DIR / "credit_scoring_model.joblib")
    feature_order = joblib.load(MODEL_DIR / "credit_scoring_features.joblib")

    df = pd.DataFrame([client])
    df[TARGET] = 0  # placeholder requis par engineer_credit_features, non utilisé
    df_fe = engineer_credit_features(df, target=TARGET).drop(columns=[TARGET])

    # Sur une seule ligne, pd.get_dummies ne crée que la colonne du bin d'âge observé ;
    # on réaligne donc sur les colonnes vues à l'entraînement (0 pour les bins absents).
    df_fe = df_fe.reindex(columns=feature_order, fill_value=0)
    return float(model.predict_proba(df_fe)[:, 1][0])


if __name__ == "__main__":
    train_all()
