"""
Module Segmentation Clients - Clustering optimisé
=================================================

Compare plusieurs algorithmes de clustering :
- KMeans
- Gaussian Mixture Model
- Agglomerative Clustering
- DBSCAN
- HDBSCAN (optionnel)

Optimisation automatique :
- nombre optimal de clusters
- hyperparamètres
- sélection du meilleur modèle

Usage:
    python src/clustering.py
"""

from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering
)

from sklearn.mixture import GaussianMixture

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")


# ==============================
# CONFIGURATION
# ==============================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR /
    "data" /
    "raw" /
    "customer_segmentation.csv"
)

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)


FEATURES = [
    "age",
    "annual_income",
    "balance",
    "spending_score",
    "tenure_years",
    "num_products"
]


# ==============================
# CHARGEMENT DES DONNEES
# ==============================

def load_data(path=DATA_PATH):

    print("Chargement des données...")

    df = pd.read_csv(path)

    print(
        f"Données chargées : {df.shape[0]} clients "
        f"x {df.shape[1]} variables"
    )

    return df

# ==============================
# PREPROCESSING
# ==============================

def preprocess(df: pd.DataFrame):

    """
    Prépare les données pour le clustering :
    - sélection des variables utiles
    - suppression des valeurs manquantes
    - normalisation robuste
    """

    X = df[FEATURES].copy()

    # Gestion des valeurs manquantes
    X = X.fillna(X.median())

    # RobustScaler résiste mieux aux valeurs extrêmes
    scaler = RobustScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler



# ==============================
# RECHERCHE DU MEILLEUR K
# ==============================

def find_best_k(X_scaled, min_k=2, max_k=10):

    """
    Recherche automatique du nombre optimal
    de clusters avec le score silhouette.
    """

    scores = []

    print("\nRecherche du meilleur nombre de clusters...")

    for k in range(min_k, max_k + 1):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )

        labels = model.fit_predict(X_scaled)

        score = silhouette_score(
            X_scaled,
            labels
        )

        scores.append(
            {
                "k": k,
                "silhouette": score
            }
        )

        print(
            f"K={k} --> silhouette={score:.4f}"
        )


    results = pd.DataFrame(scores)


    best_k = (
        results
        .sort_values(
            "silhouette",
            ascending=False
        )
        .iloc[0]["k"]
    )


    best_k = int(best_k)


    print(
        f"\nMeilleur nombre de clusters : {best_k}"
    )


    results.to_csv(
        RESULT_DIR /
        "k_selection.csv",
        index=False
    )


    return best_k

# ==============================
# ENTRAINEMENT DES MODELES
# ==============================

def train_models(X_scaled, best_k):

    """
    Entraîne plusieurs algorithmes
    avec optimisation des paramètres.
    """

    results = {}
    models = {}


    # ------------------------------
    # KMEANS
    # ------------------------------

    print("\nEntraînement KMeans...")

    best_score = -1
    best_model = None
    best_labels = None


    for init in ["k-means++", "random"]:

        for n_init in [10, 20, 50]:

            model = KMeans(
                n_clusters=best_k,
                init=init,
                n_init=n_init,
                random_state=42
            )

            labels = model.fit_predict(
                X_scaled
            )

            score = silhouette_score(
                X_scaled,
                labels
            )

            if score > best_score:
                best_score = score
                best_model = model
                best_labels = labels


    models["kmeans"] = best_model
    results["kmeans"] = best_labels



    # ------------------------------
    # GMM
    # ------------------------------

    print("Entraînement GMM...")


    best_score = -1
    best_model = None
    best_labels = None


    for covariance in [
        "full",
        "diag",
        "tied",
        "spherical"
    ]:

        model = GaussianMixture(
            n_components=best_k,
            covariance_type=covariance,
            random_state=42
        )


        labels = model.fit_predict(
            X_scaled
        )


        score = silhouette_score(
            X_scaled,
            labels
        )


        if score > best_score:

            best_score = score
            best_model = model
            best_labels = labels



    models["gmm"] = best_model
    results["gmm"] = best_labels



    # ------------------------------
    # AGGLOMERATIVE
    # ------------------------------

    print("Entraînement Agglomerative...")


    agglo = AgglomerativeClustering(
        n_clusters=best_k,
        linkage="ward"
    )


    agglo_labels = agglo.fit_predict(
        X_scaled
    )


    models["agglomerative"] = agglo
    results["agglomerative"] = agglo_labels



    # ------------------------------
    # DBSCAN
    # ------------------------------

    print("Entraînement DBSCAN...")


    best_score = -1
    best_dbscan = None
    best_labels = None


    for eps in [
        0.3,
        0.5,
        0.7,
        0.9,
        1.1
    ]:

        for min_samples in [
            5,
            10,
            20
        ]:


            model = DBSCAN(
                eps=eps,
                min_samples=min_samples
            )


            labels = model.fit_predict(
                X_scaled
            )


            # Suppression du bruit
            mask = labels != -1


            if len(
                set(labels[mask])
            ) < 2:
                continue


            score = silhouette_score(
                X_scaled[mask],
                labels[mask]
            )


            if score > best_score:

                best_score = score
                best_dbscan = model
                best_labels = labels



    models["dbscan"] = best_dbscan
    results["dbscan"] = best_labels



    return results, models

# ==============================
# EVALUATION DES MODELES
# ==============================

def evaluate_models(X_scaled, labels_dict):

    """
    Calcule les métriques :
    - Silhouette (plus haut = meilleur)
    - Davies-Bouldin (plus bas = meilleur)
    - Calinski-Harabasz (plus haut = meilleur)
    """

    rows = []


    for name, labels in labels_dict.items():

        if labels is None:
            continue


        # Suppression du bruit DBSCAN
        mask = labels != -1


        unique_clusters = len(
            set(labels[mask])
        )


        if unique_clusters < 2:

            rows.append(
                {
                    "model": name,
                    "silhouette": np.nan,
                    "davies_bouldin": np.nan,
                    "calinski_harabasz": np.nan,
                    "n_clusters": unique_clusters
                }
            )

            continue



        silhouette = silhouette_score(
            X_scaled[mask],
            labels[mask]
        )


        davies = davies_bouldin_score(
            X_scaled[mask],
            labels[mask]
        )


        calinski = calinski_harabasz_score(
            X_scaled[mask],
            labels[mask]
        )


        rows.append(
            {
                "model": name,
                "silhouette": silhouette,
                "davies_bouldin": davies,
                "calinski_harabasz": calinski,
                "n_clusters": unique_clusters
            }
        )



    df_results = (
        pd.DataFrame(rows)
        .set_index("model")
    )


    print(
        "\n=== Performance des modèles ==="
    )

    print(
        df_results.round(4)
    )


    df_results.to_csv(
        RESULT_DIR /
        "clustering_performance.csv"
    )


    return df_results



# ==============================
# CHOIX DU MEILLEUR MODELE
# ==============================

def select_best_model(
        evaluation_df
):

    """
    Sélection du meilleur modèle
    basé sur la silhouette.
    """


    best_model = (
        evaluation_df["silhouette"]
        .idxmax()
    )


    best_score = (
        evaluation_df
        .loc[
            best_model,
            "silhouette"
        ]
    )


    print(
        "\n=============================="
    )

    print(
        f"Meilleur modèle : {best_model}"
    )

    print(
        f"Score silhouette : {best_score:.4f}"
    )

    print(
        "=============================="
    )


    return best_model

# ==============================
# NOMMER LES SEGMENTS CLIENTS
# ==============================

def label_segments(df, labels):

    """
    Donne un nom métier aux clusters
    selon le profil moyen des clients.
    """

    df_result = df.copy()

    df_result["cluster"] = labels


    # Profil moyen de chaque cluster

    profile = (
        df_result
        .groupby("cluster")[FEATURES]
        .mean()
    )


    names = {}


    # Classement selon revenu + dépense

    profile["score_client"] = (
        profile["annual_income"] * 0.5
        +
        profile["spending_score"] * 0.5
    )


    ranked_clusters = (
        profile
        .sort_values(
            "score_client",
            ascending=False
        )
        .index
    )


    segment_names = [
        "Clients Premium",
        "Clients Fideles",
        "Clients Occasionnels",
        "Clients a Risque"
    ]


    for i, cluster_id in enumerate(
        ranked_clusters
    ):

        if i < len(segment_names):

            names[cluster_id] = segment_names[i]

        else:

            names[cluster_id] = (
                f"Segment {cluster_id}"
            )



    df_result["segment_name"] = (
        df_result["cluster"]
        .map(names)
    )


    return df_result



# ==============================
# SAUVEGARDE DU MODELE
# ==============================

def save_model(
        model,
        scaler,
        df_segments
):

    """
    Sauvegarde :
    - modèle clustering
    - scaler
    - résultats segmentation
    """


    model_path = (
        MODEL_DIR /
        "clustering_model.joblib"
    )


    scaler_path = (
        MODEL_DIR /
        "clustering_scaler.joblib"
    )


    data_path = (
        BASE_DIR /
        "data" /
        "customer_segments.csv"
    )


    joblib.dump(
        model,
        model_path
    )


    joblib.dump(
        scaler,
        scaler_path
    )


    df_segments.to_csv(
        data_path,
        index=False
    )


    print("\nModèle sauvegardé :")
    print(model_path)


    print("\nScaler sauvegardé :")
    print(scaler_path)


    print("\nSegments exportés :")
    print(data_path)



# ==============================
# RESUME DES SEGMENTS
# ==============================

def display_segments(df_segments):

    print(
        "\n=== Répartition des segments ==="
    )


    print(
        df_segments[
            "segment_name"
        ]
        .value_counts()
    )


    print(
        "\n=== Profil moyen des segments ==="
    )


    print(
        df_segments
        .groupby(
            "segment_name"
        )[FEATURES]
        .mean()
        .round(2)
    )

# ==============================
# VISUALISATION 2D
# ==============================

def project_2d(
        X_scaled,
        method="pca"
):

    """
    Projection des données en 2 dimensions
    pour visualisation.
    """


    if method == "pca":

        reducer = PCA(
            n_components=2
        )

        return reducer.fit_transform(
            X_scaled
        )


    elif method == "tsne":

        reducer = TSNE(
            n_components=2,
            random_state=42,
            perplexity=30
        )

        return reducer.fit_transform(
            X_scaled
        )


    else:

        raise ValueError(
            "Méthode inconnue"
        )



def save_projection(
        X_2d,
        labels
):

    """
    Sauvegarde des coordonnées 2D
    """

    df_plot = pd.DataFrame(
        X_2d,
        columns=[
            "x",
            "y"
        ]
    )


    df_plot["cluster"] = labels


    df_plot.to_csv(
        RESULT_DIR /
        "clusters_2d.csv",
        index=False
    )



# ==============================
# PIPELINE COMPLETE
# ==============================

def run_full_pipeline():

    print(
        "\n===== SEGMENTATION CLIENTS ====="
    )


    # 1 - Chargement

    df = load_data()



    # 2 - Préparation

    X_scaled, scaler = preprocess(
        df
    )



    # 3 - Recherche meilleur K

    best_k = find_best_k(
        X_scaled,
        min_k=2,
        max_k=10
    )



    # 4 - Entraînement modèles

    labels_dict, models = train_models(
        X_scaled,
        best_k
    )



    # 5 - Evaluation

    evaluation = evaluate_models(
        X_scaled,
        labels_dict
    )



    # 6 - Sélection meilleur modèle

    best_name = select_best_model(
        evaluation
    )


    best_model = models[
        best_name
    ]


    best_labels = labels_dict[
        best_name
    ]



    # 7 - Création segments

    df_segments = label_segments(
        df,
        best_labels
    )


    display_segments(
        df_segments
    )



    # 8 - Sauvegarde modèle

    save_model(
        best_model,
        scaler,
        df_segments
    )



    # 9 - Projection PCA

    X_2d = project_2d(
        X_scaled,
        "pca"
    )


    save_projection(
        X_2d,
        best_labels
    )



    print(
        "\nPipeline terminé avec succès."
    )


    return (
        df_segments,
        evaluation
    )



# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    run_full_pipeline()