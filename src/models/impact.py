# src/models/impact.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from src.data.preprocess import build_dataset
from src.models.train_model import get_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def expected_remaining_months(row: pd.Series) -> int:

    tenure = row.get("tenure", 0) or 0
    contract = row.get("Contract", "")

    if contract == "Month-to-month":
        return 6  # aanname

    if contract == "One year":
        return max(1, 12 - int(tenure))

    if contract == "Two year":
        return max(1, 24 - int(tenure))

    # fallback
    return 12


def compute_expected_loss_df(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Takes a DataFrame with customer data (optionally including target/id) and computes
    expected loss metrics based on churn probabilities and expected remaining months.
    Parameters
    ----------
    df : pd.DataFrame | None, optional
        Input DataFrame containing customer features. If None, the dataset is built
        using the `build_dataset` function. Default is None.
    """
    
    if df is None:
        df, X, y, _ = build_dataset()
    else:
        from src.data.preprocess import split_features_target
        _, X, _, _ = _split_for_predict(df)

    clf = get_model()

    if df is None:
        X_input = X
        df_scores = df.copy()
    else:
        X_input = df.copy()
        df_scores = df.copy()

    # Churn probabilities
    proba = clf.predict_proba(X_input)[:, 1]
    df_scores["churn_proba"] = proba

    # horizon per customer
    df_scores["horizon"] = df_scores.apply(expected_remaining_months, axis=1)

    # sanity check for MonthlyCharges
    if "MonthlyCharges" not in df_scores.columns:
        raise ValueError("MonthlyCharges column is required for impact calculation.")

    df_scores["expected_loss_total"] = (
        df_scores["churn_proba"] * df_scores["MonthlyCharges"] * df_scores["horizon"]
    )

    # optional: monthly risk
    df_scores["revenue_at_risk_month"] = (
        df_scores["churn_proba"] * df_scores["MonthlyCharges"]
    )

    return df_scores


# (optional) helper function if you want to score a custom df without target/id
def _split_for_predict(df: pd.DataFrame):
    """
    Ensures we build the correct X from an arbitrary df for scoring.
    Rarely used here, but you can extend it later.
    """
    from src.data.preprocess import split_features_target

    # small hack: if Churn exists, we use it as target,
    # otherwise we assume there is no target
    if "Churn" in df.columns:
        X, y, id_cols = split_features_target(df)
    else:
        id_cols = [c for c in df.columns if "customer" in c.lower() or "id" in c.lower()]
        X = df.drop(columns=id_cols, errors="ignore")
        y = None
    return df, X, y, id_cols
