import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU

from flask import Flask, request, jsonify
import torch
from src.services.sentiment_service import SentimentService

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_sentiment():
    try:
        data = request.get_json()
        text = data.get("text", "")
        result = SentimentService.predict(text)
        return jsonify({"sentiment": result})
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)

