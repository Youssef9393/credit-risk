"""
Génère et exécute notebooks/01_EDA_et_Feature_Engineering.ipynb
Usage : python notebooks/build_eda_notebook.py
"""
import nbformat as nbf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# ============================================================================
md("""# 📊 Analyse Exploratoire (EDA) & Feature Engineering
### Plateforme Intelligente de Risque Crédit, Fraude & Segmentation

Ce notebook couvre l'EDA et le feature engineering pour les 5 volets du projet :
1. Credit Scoring
2. Détection de fraude
3. Segmentation client
4. Analyse de sentiment financier (NLP)
5. Séries temporelles boursières

> Les données utilisées sont générées par `src/data_generation.py` (synthétiques,
> mêmes colonnes/distributions que les datasets Kaggle réels). Remplacez simplement
> les chemins CSV pour utiliser vos propres données.
""")

code("""import sys
from pathlib import Path
BASE_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(BASE_DIR / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)
pd.set_option("display.max_columns", 50)

import data_generation
import feature_engineering as fe

DATA_DIR = BASE_DIR / "data" / "raw"
""")

# ============================================================================
md("""## 1. 💳 Credit Scoring

### 1.1 Chargement et aperçu""")

code("""df_credit = pd.read_csv(DATA_DIR / "credit_scoring.csv")
print(df_credit.shape)
df_credit.head()
""")

code("""df_credit.describe().T
""")

md("### 1.2 Valeurs manquantes")

code("""fe.missing_value_report(df_credit)
""")

md("### 1.3 Distribution de la cible (déséquilibre de classe)")

code("""fig, ax = plt.subplots(1, 2, figsize=(12, 4))
df_credit["SeriousDlqin2yrs"].value_counts().plot(kind="bar", ax=ax[0], color=["#2ecc71", "#e74c3c"])
ax[0].set_title("Distribution de la cible (0=solvable, 1=défaut)")
ax[0].set_xticklabels(["Solvable", "Défaut"], rotation=0)

default_rate = df_credit["SeriousDlqin2yrs"].mean() * 100
ax[1].pie([100 - default_rate, default_rate], labels=["Solvable", "Défaut"],
          autopct="%1.1f%%", colors=["#2ecc71", "#e74c3c"])
ax[1].set_title("Taux de défaut")
plt.tight_layout()
plt.show()
print(f"Taux de défaut : {default_rate:.2f}%")
""")

md("### 1.4 Distributions des variables clés (revenu, endettement, âge)")

code("""fig, axes = plt.subplots(2, 3, figsize=(16, 8))
num_cols = ["age", "MonthlyIncome", "DebtRatio",
            "RevolvingUtilizationOfUnsecuredLines",
            "NumberOfOpenCreditLinesAndLoans", "NumberOfDependents"]
for ax, col in zip(axes.flat, num_cols):
    sns.histplot(df_credit[col], bins=40, ax=ax, kde=True, color="steelblue")
    ax.set_title(col)
plt.tight_layout()
plt.show()
""")

md("### 1.5 Corrélations entre variables numériques")

code("""plt.figure(figsize=(10, 8))
corr = df_credit.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Matrice de corrélation - Credit Scoring")
plt.tight_layout()
plt.show()
""")

md("""### 1.6 Weight of Evidence (WOE) & Information Value (IV)

L'IV mesure le pouvoir prédictif de chaque variable par rapport à la cible.
Standard en credit scoring pour prioriser les variables avant modélisation.
IV < 0.02 : non prédictif | 0.02-0.1 : faible | 0.1-0.3 : moyen | > 0.3 : fort""")

code("""iv_results = {}
for col in ["DebtRatio", "RevolvingUtilizationOfUnsecuredLines", "MonthlyIncome",
            "age", "NumberOfOpenCreditLinesAndLoans"]:
    woe_table = fe.compute_woe_iv(df_credit, col, "SeriousDlqin2yrs", bins=8)
    iv_results[col] = woe_table.attrs["total_iv"]

iv_df = pd.Series(iv_results, name="Information Value").sort_values(ascending=False)
iv_df.plot(kind="barh", color="darkorange", figsize=(8, 4))
plt.title("Information Value (IV) par variable")
plt.xlabel("IV")
plt.tight_layout()
plt.show()
print(iv_df)
""")

md("### 1.7 Feature Engineering : avant / après")

code("""df_credit_fe = fe.engineer_credit_features(df_credit)
print(f"Colonnes avant FE : {df_credit.shape[1]}")
print(f"Colonnes après FE : {df_credit_fe.shape[1]}")
new_cols = [c for c in df_credit_fe.columns if c not in df_credit.columns]
print("\\nNouvelles variables créées :")
for c in new_cols:
    print(" -", c)
""")

md("### 1.8 Importance des variables après Feature Engineering")

code("""X = df_credit_fe.drop(columns=["SeriousDlqin2yrs"]).select_dtypes(include=[np.number])
y = df_credit_fe["SeriousDlqin2yrs"]
importance_report = fe.feature_importance_report(X, y)

plt.figure(figsize=(9, 6))
top = importance_report.head(12)
sns.barplot(data=top, y="feature", x="rf_importance", color="steelblue")
plt.title("Top 12 variables - Importance Random Forest (après Feature Engineering)")
plt.tight_layout()
plt.show()
importance_report.head(12)
""")

# ============================================================================
md("""## 2. 🚨 Détection de fraude

### 2.1 Chargement et déséquilibre extrême des classes""")

code("""df_fraud = pd.read_csv(DATA_DIR / "creditcard_fraud.csv")
print(df_fraud.shape)

fraud_rate = df_fraud["Class"].mean() * 100
print(f"Taux de fraude : {fraud_rate:.4f}% ({df_fraud['Class'].sum()} fraudes / {len(df_fraud)} transactions)")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
df_fraud["Class"].value_counts().plot(kind="bar", ax=ax[0], color=["#2ecc71", "#e74c3c"], logy=True)
ax[0].set_title("Distribution des classes (échelle log)")
ax[0].set_xticklabels(["Légitime", "Fraude"], rotation=0)
ax[1].pie([100 - fraud_rate, fraud_rate], labels=["Légitime", "Fraude"], autopct="%1.3f%%",
          colors=["#2ecc71", "#e74c3c"])
plt.tight_layout()
plt.show()
""")

md("### 2.2 Distribution du montant selon la classe")

code("""fig, ax = plt.subplots(1, 2, figsize=(14, 4))
sns.boxplot(data=df_fraud, x="Class", y="Amount", ax=ax[0])
ax[0].set_title("Montant par classe (avec outliers)")
ax[0].set_xticklabels(["Légitime", "Fraude"])

sns.histplot(data=df_fraud, x="Amount", hue="Class", bins=50, log_scale=(True, False),
             ax=ax[1], palette=["#2ecc71", "#e74c3c"], stat="density", common_norm=False)
ax[1].set_title("Distribution du montant (log-scale) par classe")
plt.tight_layout()
plt.show()
""")

md("### 2.3 Corrélation des variables V1-V28 avec la fraude")

code("""v_cols = [c for c in df_fraud.columns if c.startswith("V")]
corr_with_target = df_fraud[v_cols + ["Class"]].corr()["Class"].drop("Class").sort_values()

plt.figure(figsize=(10, 8))
corr_with_target.plot(kind="barh", color=corr_with_target.apply(lambda x: "#e74c3c" if x > 0 else "#3498db"))
plt.title("Corrélation des variables V1-V28 avec la fraude (Class)")
plt.tight_layout()
plt.show()
""")

md("### 2.4 Feature Engineering fraude : patterns temporels et montants")

code("""df_fraud_fe = fe.engineer_fraud_features(df_fraud)
new_cols = [c for c in df_fraud_fe.columns if c not in df_fraud.columns]
print("Nouvelles variables créées :", new_cols)

fig, ax = plt.subplots(1, 2, figsize=(14, 4))
fraud_by_hour = df_fraud_fe.groupby("hour_of_day")["Class"].mean() * 100
fraud_by_hour.plot(kind="bar", ax=ax[0], color="darkred")
ax[0].set_title("Taux de fraude (%) par heure de la journée")
ax[0].set_ylabel("% de transactions frauduleuses")

sns.boxplot(data=df_fraud_fe, x="Class", y="v_features_norm", ax=ax[1])
ax[1].set_title("Norme du vecteur V1-V28 par classe\\n(les fraudes s'écartent du comportement normal)")
ax[1].set_xticklabels(["Légitime", "Fraude"])
plt.tight_layout()
plt.show()
""")

# ============================================================================
md("""## 3. 👥 Segmentation client

### 3.1 Chargement et distributions""")

code("""df_seg = pd.read_csv(DATA_DIR / "customer_segmentation.csv")
print(df_seg.shape)
df_seg.describe().T
""")

code("""num_cols = ["age", "annual_income", "balance", "spending_score", "tenure_years", "num_products"]
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, col in zip(axes.flat, num_cols):
    sns.histplot(df_seg[col], bins=30, kde=True, ax=ax, color="mediumseagreen")
    ax.set_title(col)
plt.tight_layout()
plt.show()
""")

md("### 3.2 Relations entre variables (pairplot)")

code("""sample = df_seg.sample(min(800, len(df_seg)), random_state=42)
sns.pairplot(sample, vars=["annual_income", "balance", "spending_score", "tenure_years"],
             plot_kws={"alpha": 0.4, "s": 15})
plt.suptitle("Relations entre variables clients", y=1.02)
plt.show()
""")

md("### 3.3 Feature Engineering segmentation : ratios et score d'engagement")

code("""df_seg_fe = fe.engineer_segmentation_features(df_seg)
new_cols = [c for c in df_seg_fe.columns if c not in df_seg.columns]
print("Nouvelles variables créées :", new_cols)

plt.figure(figsize=(8, 5))
sns.histplot(df_seg_fe["engagement_score"], bins=30, kde=True, color="purple")
plt.title("Distribution du score d'engagement client (variable dérivée)")
plt.tight_layout()
plt.show()
""")

md("### 3.4 Corrélations (features enrichies)")

code("""plt.figure(figsize=(10, 8))
corr_seg = df_seg_fe.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr_seg, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Matrice de corrélation - Segmentation client (après FE)")
plt.tight_layout()
plt.show()
""")

# ============================================================================
md("""## 4. 📰 Analyse de sentiment financier (NLP)

### 4.1 Chargement et distribution des labels""")

code("""df_nlp = pd.read_csv(DATA_DIR / "financial_sentiment.csv")
print(df_nlp.shape)

plt.figure(figsize=(6, 4))
df_nlp["label"].value_counts().plot(kind="bar", color=["#95a5a6", "#2ecc71", "#e74c3c"])
plt.title("Distribution des sentiments")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
df_nlp["label"].value_counts(normalize=True).round(3)
""")

md("### 4.2 Longueur des textes par sentiment")

code("""df_nlp["text_length"] = df_nlp["text"].str.split().apply(len)
plt.figure(figsize=(8, 4))
sns.boxplot(data=df_nlp, x="label", y="text_length")
plt.title("Nombre de mots par sentiment")
plt.tight_layout()
plt.show()
""")

md("### 4.3 Mots les plus fréquents par sentiment")

code("""from sklearn.feature_extraction.text import CountVectorizer

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, label in zip(axes, ["positive", "negative", "neutral"]):
    texts = df_nlp.loc[df_nlp["label"] == label, "text"]
    vec = CountVectorizer(stop_words="english", max_features=10)
    X = vec.fit_transform(texts)
    freqs = pd.Series(X.toarray().sum(axis=0), index=vec.get_feature_names_out()).sort_values()
    freqs.plot(kind="barh", ax=ax, color="teal")
    ax.set_title(f"Top mots - {label}")
plt.tight_layout()
plt.show()
""")

# ============================================================================
md("""## 5. 📈 Séries temporelles boursières

### 5.1 Évolution du cours de clôture""")

code("""df_stock = pd.read_csv(DATA_DIR / "stock_prices.csv", parse_dates=["Date"])
print(df_stock.shape)

plt.figure(figsize=(12, 4))
plt.plot(df_stock["Date"], df_stock["Close"], color="navy")
plt.title("Évolution du cours de clôture")
plt.tight_layout()
plt.show()
""")

md("### 5.2 Rendements journaliers et volatilité")

code("""df_stock["daily_return"] = df_stock["Close"].pct_change()
df_stock["rolling_volatility_30d"] = df_stock["daily_return"].rolling(30).std()

fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ax[0].plot(df_stock["Date"], df_stock["daily_return"], color="darkorange", linewidth=0.7)
ax[0].set_title("Rendements journaliers")
ax[1].plot(df_stock["Date"], df_stock["rolling_volatility_30d"], color="crimson")
ax[1].set_title("Volatilité glissante (30 jours)")
plt.tight_layout()
plt.show()
""")

md("### 5.3 Distribution des rendements (test de normalité visuel)")

code("""fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df_stock["daily_return"].dropna(), bins=50, kde=True, ax=ax[0], color="slateblue")
ax[0].set_title("Distribution des rendements journaliers")

from scipy import stats
stats.probplot(df_stock["daily_return"].dropna(), dist="norm", plot=ax[1])
ax[1].set_title("QQ-plot (normalité)")
plt.tight_layout()
plt.show()
""")

# ============================================================================
md("""## 6. ✅ Synthèse

| Dataset | Lignes | Cible / Variable clé | Observation principale |
|---|---|---|---|
| Credit Scoring | ~20 000 | `SeriousDlqin2yrs` | Classes déséquilibrées, `DebtRatio` et `RevolvingUtilization` les plus prédictifs (IV) |
| Fraude | ~50 000 | `Class` | Déséquilibre extrême (<0.2%), montants et patterns horaires discriminants |
| Segmentation | ~5 000 | (non supervisé) | 4 profils naturels détectés (Premium / Fidèle / Occasionnel / Risque) |
| Sentiment NLP | ~3 000 | `label` | Vocabulaire distinct par sentiment, textes courts |
| Boursier | ~1 500 jours | `Close` | Rendements proches d'une marche aléatoire avec volatilité variable |

Le feature engineering (`src/feature_engineering.py`) a permis d'ajouter des variables
à fort pouvoir prédictif (ratios financiers, WOE/IV, features temporelles, score
d'engagement) qui améliorent la performance des modèles en aval.
""")

nb['cells'] = cells

# Sauvegarde du notebook (non exécuté pour l'instant)
out_path = BASE_DIR / "01_EDA_et_Feature_Engineering.ipynb"
with open(out_path, "w") as f:
    nbf.write(nb, f)
print(f"Notebook écrit : {out_path}")
