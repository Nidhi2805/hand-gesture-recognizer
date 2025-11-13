# app/api.py
from flask import Flask, request, jsonify
import joblib, torch, numpy as np
from config import MODEL_DIR, MODEL_NAME, SCALER_NAME, GESTURE_LABELS

app = Flask(__name__)
scaler = joblib.load(f"{MODEL_DIR}/{SCALER_NAME}")
# load model same way as inference script (omitted here for brevity)

@app.route("/predict_landmarks", methods=["POST"])
def predict_landmarks():
    data = request.json
    lm = data.get("landmarks")  # expecting flattened list length 63
    if lm is None:
        return jsonify({"error": "no landmarks provided"}), 400
    feat = scaler.transform([lm])[0]
    # run model forward to get probs (omitted)
    # return label and confidence
    return jsonify({"label": "Open Hand", "confidence": 0.91})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
