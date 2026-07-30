"""
ML Model module for Stock Price Prediction using LSTM.
Production-safe with retry logic, fallbacks, and robust error handling.
"""
import os
import pickle
import logging
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Directory to save trained models and scalers
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

LOOKBACK_DAYS = 60
MIN_ROWS_FOR_TRAINING = 100
MIN_ROWS_FOR_FALLBACK = 60  # Below this: insufficient data error
RETRY_COUNT = 2
RETRY_DELAY_SECONDS = 2


def normalize_symbol(symbol: str) -> str:
    """Normalize symbol: uppercase and trim spaces."""
    return str(symbol).strip().upper()


def _create_ticker(symbol: str):
    """Create yfinance Ticker, using curl_cffi session if available."""
    try:
        from curl_cffi import requests
        session = requests.Session(impersonate="chrome")
        return yf.Ticker(symbol, session=session)
    except (ImportError, TypeError):
        return yf.Ticker(symbol)


def download_stock_data(symbol: str) -> pd.DataFrame:
    """
    Download historical stock data using yfinance.
    Uses period='5y' with retry logic. curl_cffi session bypasses Yahoo bot protection.
    """
    symbol = normalize_symbol(symbol)

    for attempt in range(RETRY_COUNT + 1):
        try:
            ticker = _create_ticker(symbol)
            data = ticker.history(period="5y", interval="1d", auto_adjust=True)
        except Exception as e:
            logger.warning(f"Download attempt {attempt + 1} failed for {symbol}: {e}")
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            raise ConnectionError("Network error. Please try again later.") from e

        if data is None or data.empty:
            logger.warning(f"Empty data for {symbol}, attempt {attempt + 1}/{RETRY_COUNT + 1}")
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            raise ValueError("Invalid stock symbol")

        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if "Close" not in data.columns:
            raise ValueError("Invalid stock symbol")

        data = data[["Close"]].dropna()

        if len(data) >= MIN_ROWS_FOR_FALLBACK:
            break
        elif attempt < RETRY_COUNT:
            time.sleep(RETRY_DELAY_SECONDS)
        else:
            if len(data) > 0:
                raise ValueError("Not enough historical data")
            raise ValueError("Invalid stock symbol")

    # Debug logging
    logger.info(f"Downloaded {symbol}: shape={data.shape}, rows={len(data)}")
    print(f"[DEBUG] {symbol} data shape: {data.shape}")
    print(f"[DEBUG] First rows:\n{data.head()}")
    print(f"[DEBUG] Last rows:\n{data.tail()}")

    return data


def build_model(input_shape):
    """Build LSTM model with specified architecture."""
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(64))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def get_model_path(symbol: str) -> tuple:
    """Get file paths for model and scaler."""
    symbol_clean = normalize_symbol(symbol)
    model_path = os.path.join(MODEL_DIR, f"{symbol_clean}_model.keras")
    scaler_path = os.path.join(MODEL_DIR, f"{symbol_clean}_scaler.pkl")
    return model_path, scaler_path


def train_and_save_model(symbol: str, data: pd.DataFrame, epochs: int = 15) -> tuple:
    """
    Train LSTM model on provided data, save model and scaler.
    Returns (model, scaler, data).
    """
    symbol = normalize_symbol(symbol)
    dataset = data.values

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(dataset)

    training_len = max(int(len(dataset) * 0.95), LOOKBACK_DAYS + 10)
    train_data = scaled_data[:training_len]

    x_train, y_train = [], []
    for i in range(LOOKBACK_DAYS, len(train_data)):
        x_train.append(train_data[i - LOOKBACK_DAYS : i, 0])
        y_train.append(train_data[i, 0])

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    logger.info(f"Training {symbol}: x_train shape={x_train.shape}")

    model = build_model((x_train.shape[1], 1))
    model.fit(x_train, y_train, batch_size=32, epochs=epochs, verbose=0)

    model_path, scaler_path = get_model_path(symbol)
    model.save(model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    logger.info(f"Model saved for {symbol}")
    return model, scaler, data


def load_model_and_scaler(symbol: str) -> tuple:
    """Load saved model and scaler for symbol."""
    from tensorflow.keras.models import load_model

    model_path, scaler_path = get_model_path(symbol)
    model = load_model(model_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def predict_with_fallback(symbol: str) -> dict:
    """
    Main prediction function with fallbacks.
    - Uses cached model if exists and data is fresh
    - Trains new model if sufficient data
    - Returns last close as prediction if data too small (fallback)
    """
    symbol = normalize_symbol(symbol)

    if not symbol:
        raise ValueError("Invalid stock symbol")

    # Fetch data first (validates symbol)
    try:
        data = download_stock_data(symbol)
    except ValueError as e:
        err_msg = str(e)
        if "Invalid" in err_msg:
            raise ValueError("Invalid stock symbol") from e
        raise ValueError("Not enough historical data") from e

    n_rows = len(data)
    last_date = data.index[-1].strftime("%Y-%m-%d")
    next_date = (data.index[-1] + timedelta(days=1)).strftime("%Y-%m-%d")
    current_price = float(data["Close"].iloc[-1])

    # Fallback: if too little data for LSTM training, return last close as prediction
    if n_rows < MIN_ROWS_FOR_TRAINING and n_rows >= MIN_ROWS_FOR_FALLBACK:
        logger.info(f"Fallback for {symbol}: only {n_rows} rows, using last close as prediction")
        historical = [
            {"date": idx.strftime("%Y-%m-%d"), "close": float(row["Close"])}
            for idx, row in data.tail(120).iterrows()
        ]
        return {
            "symbol": symbol,
            "predicted_price": round(current_price, 2),
            "historical_data": historical,
            "last_date": last_date,
            "prediction_date": next_date,
            "current_price": current_price,
            "fallback": True,
        }

    model_path, scaler_path = get_model_path(symbol)

    # Load cached model or train new one
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model, scaler = load_model_and_scaler(symbol)
            logger.info(f"Loaded cached model for {symbol}")
            data = download_stock_data(symbol)  # Fresh data for prediction
        except Exception as e:
            logger.warning(f"Failed to load cached model for {symbol}: {e}. Retraining.")
            model, scaler, data = train_and_save_model(symbol, data)
    else:
        model, scaler, data = train_and_save_model(symbol, data)

    # Predict
    dataset = data.values
    scaled_data = scaler.transform(dataset)
    last_60 = scaled_data[-LOOKBACK_DAYS:]
    last_60 = np.reshape(last_60, (1, LOOKBACK_DAYS, 1))

    prediction_scaled = model.predict(last_60, verbose=0)
    predicted_price = float(scaler.inverse_transform(prediction_scaled)[0, 0])

    historical = [
        {"date": idx.strftime("%Y-%m-%d"), "close": float(row["Close"])}
        for idx, row in data.tail(120).iterrows()
    ]

    return {
        "symbol": symbol,
        "predicted_price": round(predicted_price, 2),
        "historical_data": historical,
        "last_date": last_date,
        "prediction_date": next_date,
        "current_price": current_price,
        "fallback": False,
    }
