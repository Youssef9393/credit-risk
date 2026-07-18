"""
Module Détection de fraude
============================
Combine des modèles supervisés (XGBoost) et non-supervisés
(Isolation Forest, AutoEncoder) pour détecter des transactions frauduleuses
dans un jeu de données fortement déséquilibré.

Usage:
    python src/fraud_detection.py
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve, f1_score,
    confusion_matrix,
)
import xgboost as xgb

from feature_engineering import engineer_fraud_features

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard_fraud.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "Class"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Charge les données brutes et applique le pipeline de feature engineering
    (voir src/feature_engineering.py : log-amount, features temporelles, magnitude
    du vecteur PCA V1-V28, indicateurs de montant/heure inhabituels)."""
    df = pd.read_csv(path)
    return engineer_fraud_features(df)


# ---------------------------------------------------------------------------
# 1. Modèle supervisé : XGBoost (avec pondération pour classe déséquilibrée)
# ---------------------------------------------------------------------------
def train_xgboost_fraud(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr", random_state=42,
    )
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)

    print("=== XGBoost (supervisé) ===")
    print(classification_report(y_test, y_pred, digits=4))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))

    joblib.dump(model, MODEL_DIR / "fraud_xgboost.joblib")
    return model, (X_test, y_test)


# ---------------------------------------------------------------------------
# 2. Modèle non-supervisé : Isolation Forest (anomaly detection)
# ---------------------------------------------------------------------------
def train_isolation_forest(df: pd.DataFrame, contamination=0.0017):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    iso = IsolationForest(
        n_estimators=300, contamination=contamination,
        random_state=42, n_jobs=-1,
    )
    iso.fit(X)
    # -1 = anomalie -> on remappe en 1 = fraude
    y_pred = (iso.predict(X) == -1).astype(int)

    print("\n=== Isolation Forest (non-supervisé) ===")
    print(classification_report(y, y_pred, digits=4))

    joblib.dump(iso, MODEL_DIR / "fraud_isolation_forest.joblib")
    return iso


# ---------------------------------------------------------------------------
# 3. AutoEncoder (Deep Learning) : entraîné uniquement sur transactions légitimes,
#    l'erreur de reconstruction élevée signale une anomalie/fraude potentielle.
# ---------------------------------------------------------------------------
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20), nn.ReLU(),
            nn.Linear(20, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 20), nn.ReLU(),
            nn.Linear(20, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(df: pd.DataFrame, epochs=15, batch_size=256, lr=1e-3):
    X = df.drop(columns=[TARGET]).values.astype(np.float32)
    y = df[TARGET].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train = X_scaled[y == 0]  # entraîné uniquement sur transactions légitimes
    X_train_t = torch.tensor(X_train, dtype=torch.float32)

    model = AutoEncoder(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    n = X_train_t.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch = X_train_t[idx]
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"AutoEncoder epoch {epoch+1}/{epochs} - loss: {total_loss/n:.5f}")

    # Score d'anomalie = erreur de reconstruction sur tout le dataset
    with torch.no_grad():
        X_all_t = torch.tensor(X_scaled, dtype=torch.float32)
        recon_all = model(X_all_t)
        errors = ((recon_all - X_all_t) ** 2).mean(dim=1).numpy()

    threshold = np.percentile(errors[y == 0], 99)  # seuil basé sur les légitimes
    y_pred = (errors > threshold).astype(int)

    print("\n=== AutoEncoder (Deep Learning, non-supervisé) ===")
    print(classification_report(y, y_pred, digits=4))

    torch.save(model.state_dict(), MODEL_DIR / "fraud_autoencoder.pt")
    joblib.dump(scaler, MODEL_DIR / "fraud_autoencoder_scaler.joblib")
    joblib.dump(threshold, MODEL_DIR / "fraud_autoencoder_threshold.joblib")
    return model, threshold


def train_all():
    df = load_data()
    xgb_model, _ = train_xgboost_fraud(df)
    iso_model = train_isolation_forest(df)
    ae_model, threshold = train_autoencoder(df)
    return xgb_model, iso_model, ae_model


def predict_fraud_probability(transaction: dict, model=None) -> float:
    if model is None:
        model = joblib.load(MODEL_DIR / "fraud_xgboost.joblib")
    df = pd.DataFrame([transaction])
    return float(model.predict_proba(df)[:, 1][0])


if __name__ == "__main__":
    train_all()
