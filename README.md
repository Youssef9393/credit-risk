#  Plateforme Intelligente de Risque Crédit, Fraude & Segmentation Client

Complete Data Science & MLOps Solution for Banking and Finance:
An end-to-end intelligent platform integrating **Credit Scoring**, **Fraud Detection**, **Customer Segmentation**, **Financial NLP**, and **Stock Market Forecasting**, featuring interactive dashboards, RESTful APIs, automated MLOps pipelines, and containerized deployment with Docker and Kubernetes.

## Demo Video

## 🗂️ Project Structure

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

### Exemples d'appel

```bash
curl -X POST http://localhost:8000/predict/credit-scoring \
  -H "Content-Type: application/json" \
  -d '{
    "age": 40, "MonthlyIncome": 4500, "DebtRatio": 0.35,
    "NumberOfOpenCreditLinesAndLoans": 5, "NumberOfDependents": 2,
    "NumberOfTime30-59DaysPastDueNotWorse": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfTimes90DaysLate": 0,
    "RevolvingUtilizationOfUnsecuredLines": 0.4
  }'

curl -X POST http://localhost:8000/predict/fraud \
  -H "Content-Type: application/json" \
  -d '{"Amount": 250, "Time": 3600}'

curl -X POST http://localhost:8000/predict/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "Company X reports record profits this quarter"}'
```

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

## 🛠️ Stack technique

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



## Notes importantes

- Les scripts s'exécutent **immédiatement avec des données synthétiques** générées par
  `src/data_generation.py`, ce qui vous permet de valider toute la chaîne
  (entraînement → API → dashboard → Docker) sans attendre un téléchargement Kaggle.
- Une fois les vrais datasets Kaggle téléchargés, remplacez simplement les chemins
  `DATA_PATH` dans chaque module — l'architecture, les pipelines et les évaluations
  restent identiques.
- Les métriques obtenues sur données synthétiques sont volontairement modestes pour
  le credit scoring (les variables sont faiblement corrélées par construction) ;
  avec le vrai dataset "Give Me Some Credit", le ROC-AUC dépasse généralement 0.85.
