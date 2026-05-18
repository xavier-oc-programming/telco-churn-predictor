"""
app.py

Flask API and frontend for the Telco Churn Predictor.

Run with: python app.py  (development)
          gunicorn --bind=0.0.0.0 --timeout 600 app:app  (production / Azure)

Routes:
  GET  /                  → index.html demo frontend
  POST /predict           → JSON prediction with SHAP explanations
  GET  /api/features      → list of expected input features + types
  GET  /api/model-info    → most recent model_registry.csv row
"""

import joblib
import numpy as np
import pandas as pd
import shap
from flask import Flask, jsonify, render_template, request
from pathlib import Path

from shap_to_language import shap_values_to_factors

# ---------------------------------------------------------------------------
# Startup — load model artefacts once, not per request
# ---------------------------------------------------------------------------
MODEL_LOADED = False
best_model = None
scaler = None
feature_names = []
model_name = 'Unknown'
explainer = None
registry = None

try:
    best_model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    model_name = Path('models/best_model_name.txt').read_text().strip()

    # Initialise SHAP explainer once at startup (~8 s for XGBoost on first run)
    explainer = shap.TreeExplainer(best_model)

    registry_path = Path('models/model_registry.csv')
    if registry_path.exists():
        registry = pd.read_csv(registry_path)

    MODEL_LOADED = True
    print(f'Model loaded: {model_name}')
    print(f'Features: {len(feature_names)}')

except FileNotFoundError as e:
    print(f'ERROR: Model artefact missing — {e}')
    print('Run train.py first to generate models/best_model.pkl and related files.')

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Expected feature types — used by /api/features
FEATURE_TYPES = {
    'tenure': 'numeric',
    'MonthlyCharges': 'numeric',
    'TotalCharges': 'numeric',
    'SeniorCitizen': 'binary',
    'gender': 'binary',
    'Partner': 'binary_yn',
    'Dependents': 'binary_yn',
    'PhoneService': 'binary_yn',
    'MultipleLines': 'binary_yn',
    'OnlineSecurity': 'binary_yn',
    'OnlineBackup': 'binary_yn',
    'DeviceProtection': 'binary_yn',
    'TechSupport': 'binary_yn',
    'StreamingTV': 'binary_yn',
    'StreamingMovies': 'binary_yn',
    'PaperlessBilling': 'binary_yn',
    'Contract': 'categorical',
    'PaymentMethod': 'categorical',
    'InternetService': 'categorical',
}

REQUIRED_FIELDS = [
    'tenure', 'MonthlyCharges', 'Contract', 'InternetService',
    'TechSupport', 'OnlineSecurity', 'PaymentMethod', 'PaperlessBilling',
    'SeniorCitizen',
]


def preprocess_input(data: dict) -> np.ndarray:
    """
    Convert raw API input into the feature vector expected by the model.
    Mirrors the preprocessing in train.py exactly.
    """
    # Defaults for optional fields
    defaults = {
        'gender': 'Male',
        'Partner': 'No',
        'Dependents': 'No',
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'TotalCharges': float(data.get('MonthlyCharges', 0)) * float(data.get('tenure', 1)),
    }
    row = {**defaults, **data}

    # Build a one-row DataFrame so get_dummies handles encoding consistently
    df = pd.DataFrame([row])

    # Numeric conversions
    for col in ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Binary Yes/No → 1/0
    binary_yn_cols = [
        'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
    ]
    for col in binary_yn_cols:
        df[col] = (df[col] == 'Yes').astype(int)

    df['gender'] = (df['gender'] == 'Male').astype(int)

    # One-hot encode nominal categoricals (drop_first matches training)
    df = pd.get_dummies(df, columns=['Contract', 'PaymentMethod', 'InternetService'],
                        drop_first=True)

    # Align columns to training feature set
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # Scale numeric columns using the saved scaler
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df.values


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded. Run train.py first.'}), 503

    data = request.get_json(force=True)
    if not data:
        return jsonify({'error': 'Request body must be JSON.'}), 400

    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return jsonify({'error': f'Missing required fields: {missing}'}), 400

    try:
        X = preprocess_input(data)
    except Exception as e:
        return jsonify({'error': f'Preprocessing failed: {str(e)}'}), 400

    # Model prediction
    proba = float(best_model.predict_proba(X)[0, 1])

    if proba > 0.7:
        prediction = 'High Risk'
        risk_level = 'high'
    elif proba > 0.4:
        prediction = 'Medium Risk'
        risk_level = 'medium'
    else:
        prediction = 'Low Risk'
        risk_level = 'low'

    # SHAP explanation for this single prediction
    raw_shap = explainer.shap_values(X)
    if isinstance(raw_shap, list):
        shap_row = raw_shap[1][0]
    else:
        shap_row = raw_shap[0]

    risk_factors, protective_factors = shap_values_to_factors(
        shap_row.tolist(), feature_names, top_n=3
    )

    shap_dict = {feat: round(float(val), 4)
                 for feat, val in zip(feature_names, shap_row)}

    return jsonify({
        'churn_probability': round(proba, 2),
        'prediction': prediction,
        'risk_level': risk_level,
        'top_risk_factors': risk_factors,
        'top_protective_factors': protective_factors,
        'model_used': model_name,
        'shap_values_raw': shap_dict,
    })


@app.route('/api/features')
def api_features():
    return jsonify([
        {'name': name, 'type': FEATURE_TYPES.get(name, 'unknown')}
        for name in FEATURE_TYPES
    ])


@app.route('/api/model-info')
def api_model_info():
    if registry is None:
        return jsonify({'error': 'Model registry not found.'}), 503

    latest = registry.iloc[-1]
    return jsonify({
        'model_name': latest['model_name'],
        'trained_at': latest['timestamp'],
        'roc_auc': float(latest['roc_auc']),
        'f1': float(latest['f1']),
        'train_samples': int(latest['train_samples']),
        'test_samples': int(latest['test_samples']),
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
