"""
Génération de données synthétiques réalistes.

Ces fonctions servent de SUBSTITUT aux vrais datasets Kaggle
(Credit Card Fraud, Give Me Some Credit, Bank Customer Segmentation)
tant que vous n'avez pas téléchargé les fichiers réels via l'API Kaggle.

Pour utiliser les vraies données :
1. kaggle datasets download -d mlg-ulb/creditcardfraud
2. kaggle competitions download -c GiveMeSomeCredit
3. kaggle datasets download -d shivamb/bank-customer-segmentation
Placez les CSV dans data/raw/ et remplacez les appels à ces fonctions
par pd.read_csv(...).
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def generate_fraud_data(n=50_000, fraud_ratio=0.0017, seed=42) -> pd.DataFrame:
    """Simule le dataset Credit Card Fraud (features V1..V28 issues d'une PCA + Amount + Time)."""
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_ratio)
    n_legit = n - n_fraud

    legit = rng.normal(0, 1, size=(n_legit, 28))
    fraud = rng.normal(0, 1, size=(n_fraud, 28)) + rng.normal(3, 1.5, size=28)  # décalage pour rendre les fraudes séparables

    X = np.vstack([legit, fraud])
    y = np.array([0] * n_legit + [1] * n_fraud)

    amount = np.where(y == 1,
                       rng.exponential(250, size=n).clip(1, 5000),
                       rng.exponential(60, size=n).clip(1, 3000))
    time = np.sort(rng.integers(0, 172800, size=n))  # 48h en secondes

    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df["Amount"] = amount
    df["Time"] = time
    df["Class"] = y
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle
    return df


def generate_credit_scoring_data(n=20_000, seed=42) -> pd.DataFrame:
    """Simule le dataset 'Give Me Some Credit'."""
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 80, size=n)
    monthly_income = rng.lognormal(mean=8.2, sigma=0.6, size=n).clip(500, 50000)
    debt_ratio = rng.beta(2, 5, size=n)
    num_credit_lines = rng.poisson(6, size=n).clip(0, 30)
    num_dependents = rng.poisson(1, size=n).clip(0, 8)
    late_30_59 = rng.poisson(0.3, size=n).clip(0, 15)
    late_60_89 = rng.poisson(0.15, size=n).clip(0, 15)
    late_90 = rng.poisson(0.1, size=n).clip(0, 15)
    revolving_util = rng.beta(1.5, 3, size=n)

    # score de risque latent -> probabilité de défaut
    risk = (
        0.02 * (60 - age).clip(0, None)
        + 3.0 * debt_ratio
        + 1.5 * revolving_util
        + 0.6 * late_30_59
        + 0.9 * late_60_89
        + 1.3 * late_90
        - 0.00003 * monthly_income
        + rng.normal(0, 1, size=n)
    )
    prob_default = 1 / (1 + np.exp(-(risk - risk.mean()) / risk.std()))
    default = rng.binomial(1, prob_default.clip(0.01, 0.9))

    df = pd.DataFrame({
        "age": age,
        "MonthlyIncome": monthly_income.round(2),
        "DebtRatio": debt_ratio.round(4),
        "NumberOfOpenCreditLinesAndLoans": num_credit_lines,
        "NumberOfDependents": num_dependents,
        "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
        "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
        "NumberOfTimes90DaysLate": late_90,
        "RevolvingUtilizationOfUnsecuredLines": revolving_util.round(4),
        "SeriousDlqin2yrs": default,
    })
    return df


def generate_customer_segmentation_data(n=5_000, seed=42) -> pd.DataFrame:
    """Simule un dataset bancaire de segmentation client avec 4 profils latents."""
    rng = np.random.default_rng(seed)
    profiles = rng.choice(["Premium", "Fidele", "Occasionnel", "Risque"], size=n,
                           p=[0.15, 0.35, 0.35, 0.15])

    age = np.zeros(n)
    income = np.zeros(n)
    balance = np.zeros(n)
    spending_score = np.zeros(n)
    tenure_years = np.zeros(n)
    n_products = np.zeros(n)

    params = {
        "Premium":     dict(age=(45, 10), income=(15000, 4000), balance=(80000, 20000), spend=(80, 10), tenure=(10, 4), prod=(4, 1)),
        "Fidele":      dict(age=(40, 12), income=(7000, 2000),  balance=(20000, 8000),  spend=(55, 12), tenure=(8, 3),  prod=(2, 1)),
        "Occasionnel": dict(age=(30, 8),  income=(4000, 1500),  balance=(3000, 2000),   spend=(30, 15), tenure=(2, 1),  prod=(1, 1)),
        "Risque":      dict(age=(28, 9),  income=(2500, 1200),  balance=(-500, 1500),   spend=(20, 10), tenure=(1, 1),  prod=(1, 1)),
    }

    for p, cfg in params.items():
        mask = profiles == p
        k = mask.sum()
        age[mask] = rng.normal(*cfg["age"], size=k)
        income[mask] = rng.normal(*cfg["income"], size=k)
        balance[mask] = rng.normal(*cfg["balance"], size=k)
        spending_score[mask] = rng.normal(*cfg["spend"], size=k)
        tenure_years[mask] = rng.normal(*cfg["tenure"], size=k)
        n_products[mask] = rng.normal(*cfg["prod"], size=k)

    df = pd.DataFrame({
        "customer_id": [f"C{100000+i}" for i in range(n)],
        "age": age.clip(18, 90).round().astype(int),
        "annual_income": income.clip(500, None).round(2),
        "balance": balance.round(2),
        "spending_score": spending_score.clip(0, 100).round(1),
        "tenure_years": tenure_years.clip(0, 40).round(1),
        "num_products": n_products.clip(1, 8).round().astype(int),
        "true_segment": profiles,  # utilisé seulement pour validation, pas pour le clustering
    })
    return df


def generate_financial_news_sentiment(n=3000, seed=42) -> pd.DataFrame:
    """Simule le Financial PhraseBank (texte court + label de sentiment)."""
    rng = np.random.default_rng(seed)

    positive_templates = [
        "Company {c} reports record profits this quarter",
        "{c} shares surge after strong earnings beat",
        "{c} announces expansion into new markets",
        "Analysts upgrade {c} stock on strong growth outlook",
        "{c} revenue grows double digits year over year",
    ]
    negative_templates = [
        "{c} shares plunge after profit warning",
        "{c} announces massive layoffs amid falling demand",
        "{c} misses earnings expectations, stock drops",
        "Regulators fine {c} over compliance failures",
        "{c} faces bankruptcy risk as debt mounts",
    ]
    neutral_templates = [
        "{c} to hold annual shareholder meeting next month",
        "{c} appoints new chief financial officer",
        "{c} unchanged in early trading",
        "{c} releases quarterly report as scheduled",
        "Analysts maintain neutral rating on {c} stock",
    ]
    companies = ["Alpha Corp", "BetaBank", "GammaTech", "Delta Industries", "Omega Holdings",
                 "Sigma Retail", "Nexa Energy", "Vertex Motors"]

    rows = []
    for _ in range(n):
        label = rng.choice(["positive", "negative", "neutral"], p=[0.32, 0.28, 0.40])
        templ = {"positive": positive_templates, "negative": negative_templates,
                 "neutral": neutral_templates}[label]
        text = rng.choice(templ).format(c=rng.choice(companies))
        rows.append({"text": text, "label": label})
    return pd.DataFrame(rows)


def generate_stock_prices(n_days=1500, seed=42) -> pd.DataFrame:
    """Simule une série de cours boursiers (marche aléatoire géométrique + tendance + saisonnalité)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
    returns = rng.normal(0.0004, 0.018, size=n_days)
    seasonal = 0.002 * np.sin(np.arange(n_days) * 2 * np.pi / 252)
    price = 100 * np.exp(np.cumsum(returns + seasonal))
    volume = rng.integers(1_000_000, 8_000_000, size=n_days)
    df = pd.DataFrame({"Date": dates, "Close": price.round(2), "Volume": volume})
    return df


def save_all(out_dir: Path = RAW_DIR):
    generate_fraud_data().to_csv(out_dir / "creditcard_fraud.csv", index=False)
    generate_credit_scoring_data().to_csv(out_dir / "credit_scoring.csv", index=False)
    generate_customer_segmentation_data().to_csv(out_dir / "customer_segmentation.csv", index=False)
    generate_financial_news_sentiment().to_csv(out_dir / "financial_sentiment.csv", index=False)
    generate_stock_prices().to_csv(out_dir / "stock_prices.csv", index=False)
    print(f"Données synthétiques écrites dans {out_dir}")


if __name__ == "__main__":
    save_all()
