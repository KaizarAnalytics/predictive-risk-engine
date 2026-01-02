# tests/test_model.py

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "src" / "risk_engine" / "models" / "model.pkl"


def test_model_loads_successfully():
    """Model file exists and can be loaded."""
    assert MODEL_PATH.exists(), f"Model not found at {MODEL_PATH}"
    clf = joblib.load(MODEL_PATH)
    assert clf is not None


def test_model_outputs_probabilities():
    """Model outputs valid probabilities between 0 and 1."""
    clf = joblib.load(MODEL_PATH)

    # Sample input matching training schema
    df = pd.DataFrame({
        "gender": ["Female"],
        "SeniorCitizen": [0],
        "Partner": ["Yes"],
        "Dependents": ["No"],
        "tenure": [12],
        "PhoneService": ["Yes"],
        "MultipleLines": ["No"],
        "InternetService": ["Fiber optic"],
        "OnlineSecurity": ["No"],
        "OnlineBackup": ["Yes"],
        "DeviceProtection": ["No"],
        "TechSupport": ["No"],
        "StreamingTV": ["No"],
        "StreamingMovies": ["No"],
        "Contract": ["Month-to-month"],
        "PaperlessBilling": ["Yes"],
        "PaymentMethod": ["Electronic check"],
        "MonthlyCharges": [70.35],
        "TotalCharges": [844.2],
        "tenure_bucket": ["0-12"],
        "avg_monthly_total": [70.35],
        "is_month_to_month": [1],
        "has_fiber": [1],
        "multi_service": [1],
    })

    proba = clf.predict_proba(df)

    assert proba.shape == (1, 2), "Expected shape (1, 2) for binary classification"
    assert (proba >= 0).all() and (proba <= 1).all(), "Probabilities must be in [0, 1]"
    assert np.isclose(proba.sum(axis=1), 1.0).all(), "Probabilities must sum to 1"


def test_model_predict_returns_binary():
    """Model predict returns binary labels (0 or 1)."""
    clf = joblib.load(MODEL_PATH)

    df = pd.DataFrame({
        "gender": ["Male", "Female"],
        "SeniorCitizen": [1, 0],
        "Partner": ["No", "Yes"],
        "Dependents": ["No", "No"],
        "tenure": [1, 72],
        "PhoneService": ["Yes", "Yes"],
        "MultipleLines": ["No", "Yes"],
        "InternetService": ["Fiber optic", "DSL"],
        "OnlineSecurity": ["No", "Yes"],
        "OnlineBackup": ["No", "Yes"],
        "DeviceProtection": ["No", "Yes"],
        "TechSupport": ["No", "Yes"],
        "StreamingTV": ["No", "Yes"],
        "StreamingMovies": ["No", "Yes"],
        "Contract": ["Month-to-month", "Two year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": ["Electronic check", "Bank transfer (automatic)"],
        "MonthlyCharges": [95.0, 65.0],
        "TotalCharges": [95.0, 4680.0],
        "tenure_bucket": ["0-12", "49-72"],
        "avg_monthly_total": [95.0, 65.0],
        "is_month_to_month": [1, 0],
        "has_fiber": [1, 0],
        "multi_service": [1, 1],
    })

    preds = clf.predict(df)

    assert len(preds) == 2
    assert set(preds).issubset({0, 1}), "Predictions must be binary (0 or 1)"
