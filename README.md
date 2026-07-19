#  Plateforme Intelligente de Risque Crédit, Fraude & Segmentation Client

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-EC6B23?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=for-the-badge)](https://lightgbm.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![GCP](https://img.shields.io/badge/Google_Cloud-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white)](https://cloud.google.com/)

Complete Data Science & MLOps Solution for Banking and Finance:
An end-to-end intelligent platform integrating **Credit Scoring**, **Fraud Detection**, **Customer Segmentation**, **Financial NLP**, and **Stock Market Forecasting**, featuring interactive dashboards, RESTful APIs, automated MLOps pipelines, and containerized deployment with Docker and Kubernetes.

## Demo Video
<img width="1886" height="692" alt="Screenshot 2026-07-18 154101" src="https://github.com/user-attachments/assets/6052a3d2-cb46-45d7-b663-4c0a1cc0b5d4" />
<img width="1876" height="862" alt="Screenshot 2026-07-18 154132" src="https://github.com/user-attachments/assets/b81953ea-81e9-49fd-800e-2795f1878d44" />
<img width="1880" height="564" alt="Screenshot 2026-07-18 154159" src="https://github.com/user-attachments/assets/b023da4c-4deb-4c88-91d1-e82c46e083be" />
<img width="1879" height="839" alt="Screenshot 2026-07-18 154220" src="https://github.com/user-attachments/assets/bd19655d-a1ff-4a1a-8ea2-279a1f00b90a" />
<img width="1865" height="702" alt="Screenshot 2026-07-18 154434" src="https://github.com/user-attachments/assets/132308d5-c648-40a3-b07e-e55128fbca11" />





## Project Structure

```
credit-risk-platform/
├── data/
│   ├── raw/                              # Raw datasets (generated or downloaded from Kaggle)
│   └── customer_segments.csv             # Clustering output
├── notebooks/
│   ├── 01_EDA_and_Feature_Engineering.ipynb   # Complete EDA notebook (executed, with visualizations)
│   └── build_eda_notebook.py                    # Script to generate/regenerate the notebook
├── src/
│   ├── data_generation.py                # Synthetic data generator (replaces Kaggle during development)
│   ├── feature_engineering.py            # Reusable feature engineering (WOE/IV, ratios, binning, etc.)
│   ├── credit_scoring.py                 # Logistic Regression / Random Forest / XGBoost / LightGBM
│   ├── fraud_detection.py                # XGBoost + Isolation Forest + AutoEncoder (PyTorch)
│   ├── clustering.py                     # K-Means / DBSCAN / Agglomerative / GMM + PCA/t-SNE/UMAP
│   ├── nlp_sentiment.py                  # TF-IDF + Logistic Regression (baseline) + BERT fine-tuning
│   ├── time_series_lstm.py               # LSTM (PyTorch) for stock market forecasting
│   └── train_pipeline.py                 # Training orchestration + MLflow tracking
├── api/
│   └── main.py                           # FastAPI service exposing all models
├── dashboard/
│   └── app_streamlit.py                  # Interactive Streamlit dashboard
├── models/                               # Trained model artifacts (.joblib / .pt)
├── tests/
│   └── test_pipelines.py                 # Unit and integration tests (pytest)
├── .github/workflows/ci-cd.yml           # GitHub Actions CI/CD pipeline
├── Dockerfile                            # API container image
├── Dockerfile.dashboard                  # Dashboard container image
├── docker-compose.yml                    # API + Dashboard + MLflow UI stack
└── requirements.txt                      # Python dependencies
```

## Installation

```bash
git clone https://github.com/Youssef9393/credit-risk.git
cd credit-risk-platform
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data

By default, the project uses a realistic synthetic data generator (src/data_generation.py)

```bash
python src/data_generation.py
```

##  2. Analyse exploratoire (EDA) & Feature Engineering

A comprehensive and pre-executed Jupyter Notebook covers all five datasets.

```bash
jupyter notebook notebooks/01_EDA_et_Feature_Engineering.ipynb
```

Contents:
 - Data distributions, missing values, correlations, and class imbalance analysis
 - Weight of Evidence (WOE) / Information Value (IV) for credit scoring
 - Temporal patterns and transaction amount analysis for fraud detection
 - Pair plots and customer engagement score analysis for customer segmentation
 - Word frequency analysis by sentiment (NLP)
 - Returns, rolling volatility, and QQ plots for stock market time series

The src/feature_engineering.py module is directly reused by both credit_scoring.py and fraud_detection.py. As a result, the feature transformations explored in the notebook (financial ratios, log transformations, binning, temporal features, etc.) are identical to those applied during model training and in production through the API and dashboard.

## Train Models

Train an individual module:

```bash
python src/credit_scoring.py       # Credit scoring
python src/fraud_detection.py      # Détection de fraude
python src/clustering.py           # Segmentation client
python src/nlp_sentiment.py --mode baseline   # Sentiment (rapide)
python src/nlp_sentiment.py --mode bert       # Sentiment BERT
python src/time_series_lstm.py     # Prévision boursière LSTM
```

Ou entraîner **tout le pipeline avec suivi MLflow** :

```bash
python src/train_pipeline.py
mlflow ui   # http://localhost:5000
```

## Dashboard Streamlit

```bash
streamlit run dashboard/app_streamlit.py
```

Onglets disponibles : Credit Scoring, Détection de fraude, Segmentation clients,
Sentiment financier, Prévision boursière.

## API FastAPI

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
Documentation interactive : http://localhost:8000/docs

## 🐳 Conteneurisation Docker

```bash
# Construire et lancer API + Dashboard + MLflow UI ensemble
docker-compose up --build

# API      -> http://localhost:8000/docs
# Dashboard-> http://localhost:8501
# MLflow   -> http://localhost:5000
```

Build individuel :
```bash
docker build -t credit-risk-api -f Dockerfile .
docker build -t credit-risk-dashboard -f Dockerfile.dashboard .
```
## MLOps / CI-CD

- **Versionnement** : Git + GitHub
- **Suivi d'expériences** : MLflow (`mlruns/`, `mlflow ui`)
- **CI/CD** : `.github/workflows/ci-cd.yml` (tests → build Docker → push registry → déploiement)

## Tests

```bash
pytest tests/ -v
```

## Outils

| Domaine | Outils |
|---|---|
| Data | Pandas, NumPy |
| ML classique | Scikit-learn, XGBoost, LightGBM |
| Deep Learning | PyTorch (AutoEncoder, LSTM), Transformers (BERT) |
| Clustering | K-Means, DBSCAN, Agglomerative, GMM + PCA/t-SNE/UMAP |
| NLP | TF-IDF, BERT (HuggingFace) |
| Visualisation | Plotly, Matplotlib, Seaborn, Streamlit |
| API | FastAPI, Pydantic, Uvicorn |
| MLOps | MLflow, Docker, GitHub Actions|
| Cloud | GCP (Vertex AI/Cloud Run) |

## Déploiement Cloud

### AWS (EC2 / ECR / SageMaker)
```bash
# Pousser l'image vers ECR
aws ecr create-repository --repository-name credit-risk-api
docker tag credit-risk-api:latest <account_id>.dkr.ecr.<region>.amazonaws.com/credit-risk-api:latest
docker push <account_id>.dkr.ecr.<region>.amazonaws.com/credit-risk-api:latest

# Déployer sur ECS Fargate ou EC2, ou packager le modèle pour SageMaker Endpoint
```

### GCP (Cloud Run / Vertex AI)
```bash
gcloud builds submit --tag gcr.io/<project_id>/credit-risk-api
gcloud run deploy credit-risk-api \
  --image gcr.io/<project_id>/credit-risk-api \
  --platform managed --region europe-west1 --allow-unauthenticated
```

