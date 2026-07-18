"""
Pipeline d'entraînement complet avec suivi MLflow
=====================================================
Entraîne tous les modèles de la plateforme et enregistre
les métriques / artefacts dans MLflow.

Usage:
    python src/train_pipeline.py
    mlflow ui   # puis ouvrir http://localhost:5000
"""

from pathlib import Path
import mlflow
import mlflow.sklearn

import data_generation
import credit_scoring
import fraud_detection
import clustering
import nlp_sentiment
import time_series_lstm

BASE_DIR = Path(__file__).resolve().parents[1]
mlflow.set_tracking_uri(f"file://{BASE_DIR / 'mlruns'}")


def run_pipeline():
    # Étape 0 : données (si absentes)
    raw_dir = BASE_DIR / "data" / "raw"
    if not (raw_dir / "credit_scoring.csv").exists():
        print(">>> Génération des données synthétiques...")
        data_generation.save_all()

    mlflow.set_experiment("credit-risk-platform")

    # 1. Credit scoring
    with mlflow.start_run(run_name="credit_scoring"):
        best_model, results_df = credit_scoring.train_all()
        best_row = results_df.iloc[0]
        mlflow.log_params({"model_type": results_df.index[0]})
        mlflow.log_metrics(best_row.to_dict())
        mlflow.sklearn.log_model(best_model, "model")

    # 2. Détection de fraude
    with mlflow.start_run(run_name="fraud_detection"):
        df_fraud = fraud_detection.load_data()
        xgb_model, (X_test, y_test) = fraud_detection.train_xgboost_fraud(df_fraud)
        from sklearn.metrics import roc_auc_score
        proba = xgb_model.predict_proba(X_test)[:, 1]
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, proba))
        mlflow.sklearn.log_model(xgb_model, "model")
        fraud_detection.train_isolation_forest(df_fraud)
        fraud_detection.train_autoencoder(df_fraud)

    # 3. Clustering
    with mlflow.start_run(run_name="customer_segmentation"):
        df_labeled, eval_df = clustering.run_full_pipeline()
        best_algo = eval_df["silhouette"].idxmax()
        mlflow.log_params({"best_algorithm": best_algo})
        mlflow.log_metrics(eval_df.loc[best_algo].dropna().to_dict())

    # 4. NLP Sentiment
    with mlflow.start_run(run_name="nlp_sentiment"):
        model = nlp_sentiment.train_baseline()
        mlflow.sklearn.log_model(model, "model")

    # 5. LSTM prévision boursière
    with mlflow.start_run(run_name="stock_forecasting_lstm"):
        model, scaler, (y_true, y_pred) = time_series_lstm.train_lstm()
        import numpy as np
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mlflow.log_metric("mae", mean_absolute_error(y_true, y_pred))
        mlflow.log_metric("rmse", float(np.sqrt(mean_squared_error(y_true, y_pred))))

    print("\n✅ Pipeline complet terminé. Lancez `mlflow ui` pour explorer les résultats.")


if __name__ == "__main__":
    run_pipeline()
