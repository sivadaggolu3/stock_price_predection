"""
Flask API for Stock Price Prediction.
Production-safe with consistent JSON responses and robust error handling.
"""
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

from model import predict_with_fallback, normalize_symbol

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


def success_response(data: dict) -> tuple:
    """Build successful JSON response."""
    payload = {
        "status": "success",
        "symbol": data["symbol"],
        "predicted_price": data["predicted_price"],
        **{k: v for k, v in data.items() if k not in ("symbol", "predicted_price")},
    }
    return jsonify(payload), 200


def error_response(message: str, status_code: int = 400) -> tuple:
    """Build error JSON response."""
    return (
        jsonify({
            "status": "error",
            "message": message,
            "error": message,
        }),
        status_code,
    )


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "message": "Stock Prediction API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict next day stock price.
    Input:  {"symbol": "AAPL"}
    Output (success): {"status": "success", "symbol": "AAPL", "predicted_price": 192.54, ...}
    Output (error):   {"status": "error", "message": "Invalid stock symbol"}
    """
    try:
        payload = request.get_json(silent=True)
        if not payload or "symbol" not in payload:
            return error_response("Missing 'symbol' in request body", 400)

        symbol = payload["symbol"]
        if not symbol or not str(symbol).strip():
            return error_response("Invalid stock symbol", 400)

        symbol = normalize_symbol(symbol)

        result = predict_with_fallback(symbol)
        return success_response(result)

    except ValueError as e:
        err_msg = str(e).strip()
        if "Invalid stock symbol" in err_msg:
            return error_response("Invalid stock symbol", 400)
        if "Not enough" in err_msg or "Insufficient" in err_msg:
            return error_response("Not enough historical data", 400)
        return error_response(err_msg or "Invalid request", 400)

    except ConnectionError as e:
        logger.exception("Network error during prediction")
        return error_response(str(e) or "Network error", 503)

    except Exception as e:
        logger.exception("Unexpected error during prediction")
        return error_response(f"Prediction failed: {str(e)}", 500)


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Stock Prediction API on port {port}")
    app.run(host="0.0.0.0", port=port)