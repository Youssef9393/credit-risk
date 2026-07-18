"""
Module Séries temporelles - Prévision boursière avec LSTM
==============================================================
Entraîne un réseau LSTM pour prédire le prix de clôture (Close)
à J+1 à partir d'une fenêtre glissante de prix passés.

Usage:
    python src/time_series_lstm.py
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "stock_prices.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

WINDOW_SIZE = 30


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def make_sequences(prices: np.ndarray, window=WINDOW_SIZE):
    X, y = [], []
    for i in range(len(prices) - window):
        X.append(prices[i:i + window])
        y.append(prices[i + window])
    return np.array(X), np.array(y)


class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)


def train_lstm(epochs=40, batch_size=32, lr=1e-3, test_ratio=0.15):
    df = load_data()
    prices = df["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices).flatten()

    X, y = make_sequences(prices_scaled, WINDOW_SIZE)
    split = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

    model = LSTMForecaster()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    n = X_train_t.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - train MSE: {total_loss/n:.6f}")

    # Évaluation
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_t).numpy().flatten()

    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n=== LSTM - Évaluation sur le test set ===\nMAE: {mae:.3f}  RMSE: {rmse:.3f}")

    torch.save(model.state_dict(), MODEL_DIR / "lstm_stock_forecaster.pt")
    joblib.dump(scaler, MODEL_DIR / "lstm_scaler.joblib")

    return model, scaler, (y_true, y_pred)


def forecast_next_days(model, scaler, last_window: np.ndarray, n_days=5):
    """Prévision itérative sur n_days jours à partir de la dernière fenêtre observée."""
    model.eval()
    window = scaler.transform(last_window.reshape(-1, 1)).flatten().tolist()
    preds = []
    with torch.no_grad():
        for _ in range(n_days):
            x = torch.tensor(window[-WINDOW_SIZE:], dtype=torch.float32).view(1, WINDOW_SIZE, 1)
            next_scaled = model(x).item()
            preds.append(next_scaled)
            window.append(next_scaled)
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()


if __name__ == "__main__":
    train_lstm()
