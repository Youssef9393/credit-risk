"""
API FastAPI - Plateforme Intelligente de Risque Crédit
===========================================================
Lancer avec :
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Documentation interactive : http://localhost:8000/docs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

app = FastAPI(
    title="Credit Risk & Fraud Detection Platform API",
    description="API pour le scoring crédit, la détection de fraude, la segmentation client et l'analyse de sentiment",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Modèles Pydantic (schémas de requête)
# ---------------------------------------------------------------------------
class CreditScoringRequest(BaseModel):
    age: int = Field(..., ge=18, le=100)
    MonthlyIncome: float = Field(..., ge=0)
    DebtRatio: float = Field(..., ge=0)
    NumberOfOpenCreditLinesAndLoans: int = Field(..., ge=0)
    NumberOfDependents: int = Field(0, ge=0)
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(0, ge=0, alias="NumberOfTime30-59DaysPastDueNotWorse")
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(0, ge=0, alias="NumberOfTime60-89DaysPastDueNotWorse")
    NumberOfTimes90DaysLate: int = Field(0, ge=0)
    RevolvingUtilizationOfUnsecuredLines: float = Field(..., ge=0)

    class Config:
        populate_by_name = True


class FraudRequest(BaseModel):
    Amount: float = Field(..., ge=0)
    Time: float = Field(..., ge=0)
    features: dict = Field(default_factory=dict, description="V1..V28 optionnels")


class SentimentRequest(BaseModel):
    text: str


class CreditScoringResponse(BaseModel):
    default_probability: float
    risk_level: str


class FraudResponse(BaseModel):
    fraud_probability: float
    is_suspicious: bool


class SentimentResponse(BaseModel):
    sentiment: str


# ---------------------------------------------------------------------------
# Chargement paresseux des modèles (cache en mémoire)
# ---------------------------------------------------------------------------
_models_cache = {}


def get_model(name: str, filename: str):
    if name not in _models_cache:
        path = MODEL_DIR / filename
        if not path.exists():
            raise HTTPException(status_code=503, detail=f"Modèle '{name}' non entraîné ({filename} introuvable)")
        _models_cache[name] = joblib.load(path)
    return _models_cache[name]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Credit Risk & Fraud Detection Platform API"}


@app.get("/health")
def health():
    status = {}
    for name, fname in [
        ("credit_scoring", "credit_scoring_model.joblib"),
        ("fraud_xgboost", "fraud_xgboost.joblib"),
        ("clustering", "clustering_model.joblib"),
        ("nlp_sentiment", "nlp_sentiment_baseline.joblib"),
    ]:
        status[name] = (MODEL_DIR / fname).exists()
    return status


@app.post("/predict/credit-scoring", response_model=CreditScoringResponse)
def predict_credit_scoring(req: CreditScoringRequest):
    from feature_engineering import engineer_credit_features

    model = get_model("credit_scoring", "credit_scoring_model.joblib")
    feature_order = get_model("credit_scoring_features", "credit_scoring_features.joblib")

    data = req.dict(by_alias=True)
    df = pd.DataFrame([data])
    df["SeriousDlqin2yrs"] = 0  # placeholder requis par la fonction, non utilisé
    df_fe = engineer_credit_features(df, target="SeriousDlqin2yrs").drop(columns=["SeriousDlqin2yrs"])
    df_fe = df_fe.reindex(columns=feature_order, fill_value=0)

    proba = float(model.predict_proba(df_fe)[:, 1][0])
    risk_level = "low" if proba < 0.3 else "medium" if proba < 0.6 else "high"
    return CreditScoringResponse(default_probability=round(proba, 4), risk_level=risk_level)


@app.post("/predict/fraud", response_model=FraudResponse)
def predict_fraud(req: FraudRequest):
    from feature_engineering import engineer_fraud_features

    model = get_model("fraud_xgboost", "fraud_xgboost.joblib")

    rng = np.random.default_rng(0)
    v_features = {f"V{i}": req.features.get(f"V{i}", float(rng.normal(0, 1))) for i in range(1, 29)}
    transaction = {**v_features, "Amount": req.Amount, "Time": req.Time}

    df = engineer_fraud_features(pd.DataFrame([transaction]))
    df = df[model.get_booster().feature_names]  # réaligne l'ordre des colonnes vues à l'entraînement

    proba = float(model.predict_proba(df)[:, 1][0])
    return FraudResponse(fraud_probability=round(proba, 4), is_suspicious=proba > 0.5)


@app.post("/predict/sentiment", response_model=SentimentResponse)
def predict_sentiment(req: SentimentRequest):
    model = get_model("nlp_sentiment", "nlp_sentiment_baseline.joblib")
    sentiment = model.predict([req.text])[0]
    return SentimentResponse(sentiment=sentiment)


@app.get("/segments/summary")
def segments_summary():
    seg_path = BASE_DIR / "data" / "customer_segments.csv"
    if not seg_path.exists():
        raise HTTPException(status_code=503, detail="Segmentation non calculée")
    df = pd.read_csv(seg_path)
    return df["segment_name"].value_counts().to_dict()
