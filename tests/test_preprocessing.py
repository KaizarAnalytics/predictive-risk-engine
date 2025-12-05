# tests/test_preprocessing.py

import pandas as pd
import numpy as np

from src.data.feature_engineering import apply_feature_engineering
from src.data.preprocess import build_dataset


def test_apply_feature_engineering_adds_expected_columns():
    """
    Check that apply_feature_engineering adds the expected new columns.
    """
    df = pd.DataFrame({
        "tenure": [1, 15, 40, 80],
        "MonthlyCharges": [50.0, 70.0, 90.0, 110.0],
        "TotalCharges": [50.0, 1050.0, 3600.0, 8800.0],
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
        "InternetService": ["Fiber optic", "DSL", "No", "Fiber optic"],
        "PhoneService": ["Yes", "Yes", "No", "Yes"],
    })

    out = apply_feature_engineering(df)

    # Expected new columns
    expected_cols = [
        "tenure_bucket",
        "avg_monthly_total",
        "is_month_to_month",
        "has_fiber",
        "multi_service",
    ]

    for col in expected_cols:
        assert col in out.columns, f"Feature engineering mist kolom: {col}"


def test_apply_feature_engineering_does_not_modify_input_inplace():
    """
    Ensure that the original DataFrame is not modified in-place.
    (Important to avoid nasty side effects.)
    """
    df = pd.DataFrame({
        "tenure": [5],
        "MonthlyCharges": [70.0],
        "TotalCharges": [350.0],
        "Contract": ["Month-to-month"],
        "InternetService": ["Fiber optic"],
        "PhoneService": ["Yes"],
    })

    df_copy = df.copy(deep=True)
    _ = apply_feature_engineering(df)

    # DataFrame content should remain the same
    pd.testing.assert_frame_equal(df, df_copy)


def test_tenure_bucket_has_expected_categories():
    """
    Check that tenure_bucket has the expected categories and that the mapping is logical.
    """
    df = pd.DataFrame({
        "tenure": [1, 12, 13, 24, 25, 48, 49, 72, 73],
        "MonthlyCharges": [50.0] * 9,
        "TotalCharges": [50.0] * 9,
        "Contract": ["Month-to-month"] * 9,
        "InternetService": ["DSL"] * 9,
        "PhoneService": ["Yes"] * 9,
    })

    out = apply_feature_engineering(df)

    assert "tenure_bucket" in out.columns

    # Expected labels from your FE function
    expected_categories = ["0-12", "13-24", "25-48", "49-72", "72+"]

    # All values should fall into the expected categories
    buckets = out["tenure_bucket"].astype(str).unique().tolist()
    for b in buckets:
        assert b in expected_categories, f"Unexpected tenure_bucket: {b}"

def test_build_dataset_returns_consistent_shapes():
    """
    E2E sanity check: build_dataset must return a valid (df, X, y, id_cols).
    - df and X have the same number of rows
    - y has the same length
    - there is at least 1 feature column
    """
    df, X, y, id_cols = build_dataset()

    # Basic checks
    assert len(df) > 0, "Dataset is empty"
    assert len(X) == len(df), "X and df have different lengths"
    assert len(y) == len(df), "y and df have different lengths"

    # At least 1 column in X
    assert X.shape[1] > 0, "X has no feature columns"

    # Target must be binary (0/1)
    unique_y = sorted(y.dropna().unique().tolist())
    assert set(unique_y).issubset({0, 1}), f"Target does not contain binary labels: {unique_y}"