# 🏦 Plateforme Intelligente de Risque Crédit, Fraude & Segmentation Client

Solution complète de Data Science / MLOps appliquée à la banque et la finance :
**Credit Scoring**, **détection de fraude**, **segmentation client**, **NLP financier**
et **prévision boursière**, avec dashboard, API et déploiement conteneurisé.

## 🗂️ Structure du projet

```
credit-risk-platform/
├── data/
│   ├── raw/                       # Données brutes (générées ou téléchargées de Kaggle)
│   └── customer_segments.csv      # Sortie du clustering
├── notebooks/
│   ├── 01_EDA_et_Feature_Engineering.ipynb   # Notebook EDA complet (exécuté, avec graphiques)
│   └── build_eda_notebook.py                 # Script qui génère/régénère le notebook
├── src/
│   ├── data_generation.py         # Générateur de données synthétiques (remplace Kaggle en dev)
│   ├── feature_engineering.py     # Feature engineering réutilisable (WOE/IV, ratios, binning...)
│   ├── credit_scoring.py          # LogReg / RF / XGBoost / LightGBM
│   ├── fraud_detection.py         # XGBoost + Isolation Forest + AutoEncoder (PyTorch)
│   ├── clustering.py              # K-Means / DBSCAN / Agglomerative / GMM + PCA/t-SNE/UMAP
│   ├── nlp_sentiment.py           # TF-IDF+LogReg (baseline) + BERT fine-tuning
│   ├── time_series_lstm.py        # LSTM (PyTorch) pour prévision boursière
│   └── train_pipeline.py          # Orchestration + tracking MLflow
├── api/
│   └── main.py                    # API FastAPI (tous les modèles)
├── dashboard/
│   └── app_streamlit.py           # Dashboard interactif Streamlit
├── models/                        # Modèles entraînés (.joblib / .pt)
├── tests/
│   └── test_pipelines.py          # Tests unitaires (pytest)
├── .github/workflows/ci-cd.yml    # CI/CD GitHub Actions
├── Dockerfile                     # Image API
├── Dockerfile.dashboard           # Image Dashboard
├── docker-compose.yml             # API + Dashboard + MLflow UI
└── requirements.txt
```

## 🚀 Installation

```bash
git clone <votre-repo>
cd credit-risk-platform
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 📊 1. Données

Par défaut, le projet utilise un **générateur de données synthétiques** réaliste
(`src/data_generation.py`) pour que tout soit exécutable sans compte Kaggle.

```bash
python src/data_generation.py
```

### Pour utiliser les vrais datasets Kaggle

```bash
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
kaggle competitions download -c GiveMeSomeCredit -p data/raw --unzip
kaggle datasets download -d shivamb/bank-customer-segmentation -p data/raw --unzip
```

Puis renommez/adaptez les colonnes dans `src/*.py` (`DATA_PATH`, `TARGET`) pour
pointer vers les fichiers réels — la logique de modélisation ne change pas.

## 🔍 2. Analyse exploratoire (EDA) & Feature Engineering

Un notebook Jupyter complet et **déjà exécuté** couvre les 5 datasets :

```bash
jupyter notebook notebooks/01_EDA_et_Feature_Engineering.ipynb
```

Contenu :
- Distributions, valeurs manquantes, corrélations, déséquilibre de classes
- **Weight of Evidence (WOE) / Information Value (IV)** pour le credit scoring
- Patterns temporels et montants pour la détection de fraude
- Pairplots et score d'engagement pour la segmentation client
- Fréquence de mots par sentiment (NLP)
- Rendements, volatilité glissante et QQ-plot pour la série boursière

Pour régénérer le notebook après modification des données :
```bash
python notebooks/build_eda_notebook.py   # reconstruit le notebook (cellules)
jupyter nbconvert --to notebook --execute --inplace notebooks/01_EDA_et_Feature_Engineering.ipynb
```

Le module `src/feature_engineering.py` est **réutilisé directement** par
`credit_scoring.py` et `fraud_detection.py` — les transformations vues dans le
notebook (ratios financiers, log-transform, binning, features temporelles...)
sont donc exactement celles utilisées à l'entraînement et en production (API/dashboard).

## 🧠 3. Entraînement des modèles

Entraîner un module individuellement :

```bash
python src/credit_scoring.py       # Credit scoring
python src/fraud_detection.py      # Détection de fraude
python src/clustering.py           # Segmentation client
python src/nlp_sentiment.py --mode baseline   # Sentiment (rapide)
python src/nlp_sentiment.py --mode bert       # Sentiment (BERT, plus lent, GPU recommandé)
python src/time_series_lstm.py     # Prévision boursière LSTM
```

Ou entraîner **tout le pipeline avec suivi MLflow** :

```bash
python src/train_pipeline.py
mlflow ui   # http://localhost:5000
```

## 📈 4. Dashboard Streamlit

```bash
streamlit run dashboard/app_streamlit.py
```

Onglets disponibles : Credit Scoring, Détection de fraude, Segmentation clients,
Sentiment financier, Prévision boursière.

## 🌐 5. API FastAPI

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

## 🐳 6. Conteneurisation (Docker)

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

## ☁️ 7. Déploiement Cloud

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

## 🔁 8. MLOps / CI-CD

- **Versionnement** : Git + GitHub
- **Suivi d'expériences** : MLflow (`mlruns/`, `mlflow ui`)
- **CI/CD** : `.github/workflows/ci-cd.yml` (tests → build Docker → push registry → déploiement)
- **Orchestration avancée (optionnel)** : Kubeflow Pipelines pour enchaîner
  data prep → training → evaluation → deployment sur Kubernetes

## 🧪 9. Tests

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
| MLOps | MLflow, Docker, GitHub Actions, (Kubeflow en option) |
| Cloud | AWS (EC2/S3/SageMaker/ECR) ou GCP (Vertex AI/Cloud Run) |

## ⚠️ Notes importantes

- Les scripts s'exécutent **immédiatement avec des données synthétiques** générées par
  `src/data_generation.py`, ce qui vous permet de valider toute la chaîne
  (entraînement → API → dashboard → Docker) sans attendre un téléchargement Kaggle.
- Une fois les vrais datasets Kaggle téléchargés, remplacez simplement les chemins
  `DATA_PATH` dans chaque module — l'architecture, les pipelines et les évaluations
  restent identiques.
- Les métriques obtenues sur données synthétiques sont volontairement modestes pour
  le credit scoring (les variables sont faiblement corrélées par construction) ;
  avec le vrai dataset "Give Me Some Credit", le ROC-AUC dépasse généralement 0.85.
