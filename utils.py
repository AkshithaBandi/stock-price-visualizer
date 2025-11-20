import os

import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# optional: silence TF logs (put before importing tensorflow if you import TF)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def get_stock_data(ticker, period="1y"):
    """
    Returns historical DataFrame (index = DatetimeIndex) or None on failure.
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def get_current_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1d", interval="1m")
        if df is None or df.empty:
            # fallback: use 1d daily
            df = stock.history(period="1d")
            if df is None or df.empty:
                return None
        price = float(df["Close"].iloc[-1])
        return price
    except Exception:
        return None


def moving_averages(df):
    df = df.copy()
    if "Close" not in df.columns:
        return df
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    return df


# ----------------------
# LSTM predictor
# ----------------------
def lstm_predict_close(ticker, look_back=50, days=1, epochs=2, return_series=False):
    """
    Train a tiny LSTM on closing prices and forecast `days` future closes.
    - look_back: number of past timesteps used as input
    - days: how many future days to predict
    - epochs: small by default for interactivity
    Returns: list of floats (predicted closes) or numpy array.
    Note: This function trains a small model locally — keep epochs low.
    """
    try:
        import tensorflow as tf
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Dropout
    except Exception as e:
        raise RuntimeError("TensorFlow is required for LSTM prediction. Install tensorflow.") from e

    df = get_stock_data(ticker, period="2y")
    if df is None or df.empty or "Close" not in df.columns:
        return None

    series = df["Close"].astype(np.float32).values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)

    # create sequences
    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i - look_back : i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)

    if len(X) == 0:
        return None

    X = X.reshape((X.shape[0], X.shape[1], 1))

    # tiny model for interactivity
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.1))
    model.add(LSTM(16, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    # Train (quiet)
    model.fit(X, y, epochs=max(1, int(epochs)), batch_size=32, verbose=0)

    # forecast `days`
    last_window = scaled[-look_back:].reshape(1, look_back, 1)
    preds_scaled = []
    for _ in range(days):
        p = model.predict(last_window, verbose=0)[0][0]
        preds_scaled.append(p)
        # append p and roll window
        last_window = np.concatenate([last_window[:, 1:, :], np.array(p).reshape(1, 1, 1)], axis=1)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten().tolist()
    if return_series:
        # return the historical close + predicted appended
        historical = series.flatten().tolist()
        return historical + preds
    return preds
