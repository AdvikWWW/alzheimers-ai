
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("alzheimers_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("features")
    if not data or len(data) != 40:
        return jsonify({"error": "Invalid input. Must provide 40 features."}), 400

    features = np.array(data).reshape(1, -1)
    prediction = model.predict(features)[0]
    label = label_encoder.inverse_transform([prediction])[0]
    proba = model.predict_proba(features)[0][prediction] * 100

    return jsonify({
        "prediction": label,
        "confidence": round(proba, 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
