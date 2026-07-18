"""
Module NLP - Analyse de sentiment des actualités financières
=================================================================
Deux approches :
 1. Baseline rapide : TF-IDF + Logistic Regression
 2. Modèle avancé   : fine-tuning d'un BERT (via HuggingFace transformers)

Usage:
    python src/nlp_sentiment.py --mode baseline
    python src/nlp_sentiment.py --mode bert
"""

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "financial_sentiment.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

LABELS = ["negative", "neutral", "positive"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# 1. Baseline TF-IDF + Logistic Regression (rapide, bon pour prototypage/CI)
# ---------------------------------------------------------------------------
def train_baseline():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    print("=== Baseline TF-IDF + Logistic Regression ===")
    print(classification_report(y_test, y_pred, digits=4))

    joblib.dump(pipe, MODEL_DIR / "nlp_sentiment_baseline.joblib")
    return pipe


def predict_sentiment_baseline(text: str, model=None) -> str:
    if model is None:
        model = joblib.load(MODEL_DIR / "nlp_sentiment_baseline.joblib")
    return model.predict([text])[0]


# ---------------------------------------------------------------------------
# 2. Fine-tuning BERT (transformers) - plus lourd, meilleure précision réelle
# ---------------------------------------------------------------------------
def train_bert(epochs=3, batch_size=16, model_name="distilbert-base-uncased"):
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    df = load_data()
    label2id = {l: i for i, l in enumerate(LABELS)}
    df["label_id"] = df["label"].map(label2id)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label_id"], test_size=0.2, stratify=df["label_id"], random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class SentimentDataset(Dataset):
        def __init__(self, texts, labels):
            self.texts = texts.reset_index(drop=True)
            self.labels = labels.reset_index(drop=True)

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = tokenizer(self.texts[idx], truncation=True, padding="max_length",
                             max_length=64, return_tensors="pt")
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    train_ds = SentimentDataset(X_train, y_train)
    test_ds = SentimentDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS)
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - loss: {total_loss/len(train_loader):.4f}")

    # Évaluation
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            trues.extend(labels.numpy())

    id2label = {i: l for l, i in label2id.items()}
    print("=== BERT fine-tuné ===")
    print(classification_report(
        [id2label[t] for t in trues], [id2label[p] for p in preds], digits=4
    ))

    save_dir = MODEL_DIR / "nlp_sentiment_bert"
    save_dir.mkdir(exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "bert"], default="baseline")
    args = parser.parse_args()

    if args.mode == "baseline":
        train_baseline()
    else:
        train_bert()
