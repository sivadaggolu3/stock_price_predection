# Stock Price Prediction - Full Stack App

A full-stack application that predicts next-day stock prices using a TensorFlow LSTM model. Built with Flask (backend) and vanilla HTML/CSS/JS + Bootstrap + Chart.js (frontend).

## Project Structure

```
stock_prediction_app/
├── backend/
│   ├── app.py              # Flask API server
│   ├── model.py            # LSTM model & prediction logic
│   └── saved_models/       # Cached models (created on first run)
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── requirements.txt
└── README.md
```

## How to Run

### Step 1: Create a virtual environment (recommended)

```bash
cd stock_prediction_app
python -m venv venv
```

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Start the backend

```bash
cd backend
python app.py
```

You should see:
```
Starting Stock Prediction API on http://localhost:5000
 * Running on http://0.0.0.0:5000
```

### Step 4: Open the frontend

Open `frontend/index.html` in your browser in one of these ways:

**Option A – Direct file:** Double-click `frontend/index.html` or drag it into the browser.

**Option B – Simple HTTP server (avoids CORS issues):**
```bash
cd frontend
python -m http.server 8080
```
Then open http://localhost:8080 in your browser.

### Step 5: Use the app

1. Enter a stock symbol (e.g., AAPL, GOOG, MSFT, AMZN).
2. Click **Predict Price**.
3. Wait for training (first request) or prediction (if model is cached).
4. View the predicted price and chart.

---

## API Reference

### POST /predict

**Request:**
```json
{
  "symbol": "AAPL"
}
```

**Response:**
```json
{
  "symbol": "AAPL",
  "predicted_price": 192.54,
  "current_price": 190.25,
  "historical_data": [{"date": "2024-01-01", "close": 185.0}, ...],
  "last_date": "2024-03-19",
  "prediction_date": "2024-03-20"
}
```

---

## Notes

- **First request per symbol:** The model is trained on ~12 years of data. This can take 1–3 minutes.
- **Later requests:** The trained model is cached in `backend/saved_models/`. Subsequent requests for the same symbol are faster.
- Use valid ticker symbols (e.g., AAPL, GOOG, MSFT, TSLA, AMZN).
