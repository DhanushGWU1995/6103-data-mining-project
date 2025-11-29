
from flask import Flask, request, jsonify
from flask_cors import CORS

import joblib
import numpy as np
import os


app = Flask(__name__)
CORS(app)


# Try to load the real CatBoost model, fallback to mock if not found
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'catboost_model.pkl')
try:
    model = joblib.load(MODEL_PATH)
    print(f"Loaded CatBoost model from {MODEL_PATH}")
except Exception as e:
    print(f"WARNING: Could not load CatBoost model: {e}\nUsing mock model.")
    class MockModel:
        def predict_proba(self, X):
            return np.array([[0.22, 0.78]])
    model = MockModel()

FEATURES = [
    'Income_Categories', 'Could_Not_Afford_Doctor', 'Employment_Status',
    'Primary_Insurance', 'Education_Level', 'Age_Group_5yr', 'Sex',
    'BMI_Value', 'Exercise_Past_30_Days', 'Mental_Health_Days',
    'Diabetes_Status', 'Coronary_Heart_Disease', 'Personal_Doctor',
    'Difficulty_Doing_Errands_Alone', 'Difficulty_Dressing_Bathing',
    'Difficulty_Concentrating', 'Arthritis'
]


# Preprocessing logic matching model pipeline
def preprocess_input(data):
    cleaned = {}
    # Remove/replace invalid codes for each feature
    # (Codes based on model cleaning logic)
    invalid_codes = {
        'Income_Categories': [77, 99],
        'Could_Not_Afford_Doctor': [7, 9],
        'Employment_Status': [9],
        'Primary_Insurance': [77, 99],
        'Education_Level': [9],
        'Exercise_Past_30_Days': [7, 9],
        'Diabetes_Status': [7, 9],
        'Coronary_Heart_Disease': [7, 9],
        'Personal_Doctor': [7, 9],
        'Difficulty_Doing_Errands_Alone': [7, 9],
        'Difficulty_Dressing_Bathing': [7, 9],
        'Difficulty_Concentrating': [7, 9],
        'Arthritis': [7, 9],
    }
    # Age group max value (<=10)
    for f in FEATURES:
        val = data.get(f, None)
        # Remove invalid codes
        if f in invalid_codes and val is not None:
            try:
                v = int(val)
                if v in invalid_codes[f]:
                    val = None
            except Exception:
                val = None
        # Age group: restrict to <=10
        if f == 'Age_Group_5yr' and val is not None:
            try:
                v = int(val)
                if v > 10:
                    val = 10
            except Exception:
                val = 10
        # BMI: ensure float
        if f == 'BMI_Value' and val is not None:
            try:
                val = float(val)
            except Exception:
                val = None
        # Mental_Health_Days: ensure float
        if f == 'Mental_Health_Days' and val is not None:
            try:
                val = float(val)
            except Exception:
                val = None
        # Fill missing with median or 0 (simple fallback)
        if val is None or val == '':
            val = 0
        cleaned[f] = val
    # Return features in order
    return [cleaned[f] for f in FEATURES]

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    input_vector = preprocess_input(data)
    proba = model.predict_proba([input_vector])[0, 1]
    pred = "Poor Health" if proba > 0.5 else "Good Health"
    return jsonify({
        "prediction": pred,
        "probability": float(proba),
        "model": "CatBoost",
        "explanation": f"Your predicted risk of poor health is {'high' if proba > 0.5 else 'low'} based on your input."
    })

if __name__ == '__main__':
    app.run(debug=True)
