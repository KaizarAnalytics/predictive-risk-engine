"""
Model evaluation utilities.

Provides standard metrics for binary classification and
lift/gain analysis for churn prediction.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from typing import Dict, Tuple


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
) -> Dict[str, float]:
    """
    Compute standard classification metrics.

    Args:
        y_true: True binary labels (0/1)
        y_pred: Predicted binary labels (0/1)
        y_proba: Predicted probabilities for positive class (optional)

    Returns:
        Dictionary with accuracy, precision, recall, f1, and optionally AUC/PR-AUC
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["pr_auc"] = average_precision_score(y_true, y_proba)

    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """
    Compute confusion matrix as a labeled DataFrame.

    Returns:
        DataFrame with TN, FP, FN, TP labeled
    """
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(
        cm,
        index=["Actual: No Churn", "Actual: Churn"],
        columns=["Pred: No Churn", "Pred: Churn"],
    )


def compute_lift_table(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute lift/gain table by probability deciles.

    Ranks customers by predicted churn probability and computes
    cumulative capture rate vs random baseline.

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        n_bins: Number of bins (default 10 for deciles)

    Returns:
        DataFrame with lift metrics per bin
    """
    df = pd.DataFrame({"y_true": y_true, "y_proba": y_proba})
    df = df.sort_values("y_proba", ascending=False).reset_index(drop=True)

    # Assign decile bins (1 = highest risk)
    df["decile"] = pd.qcut(
        df.index, q=n_bins, labels=range(1, n_bins + 1), duplicates="drop"
    )

    # Aggregate by decile
    lift_df = (
        df.groupby("decile", observed=True)
        .agg(
            n_customers=("y_true", "count"),
            n_churners=("y_true", "sum"),
            avg_proba=("y_proba", "mean"),
        )
        .reset_index()
    )

    # Compute rates
    total_churners = df["y_true"].sum()
    total_customers = len(df)
    base_rate = total_churners / total_customers

    lift_df["churn_rate"] = lift_df["n_churners"] / lift_df["n_customers"]
    lift_df["lift"] = lift_df["churn_rate"] / base_rate

    # Cumulative metrics
    lift_df["cum_customers"] = lift_df["n_customers"].cumsum()
    lift_df["cum_churners"] = lift_df["n_churners"].cumsum()
    lift_df["cum_churn_captured"] = lift_df["cum_churners"] / total_churners
    lift_df["cum_customers_pct"] = lift_df["cum_customers"] / total_customers

    return lift_df


def compute_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """
    Find the optimal probability threshold for a given metric.

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'recall', 'precision')

    Returns:
        Tuple of (optimal_threshold, metric_value)
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_score = 0.0

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)

        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold, best_score


def print_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
) -> None:
    """
    Print a comprehensive evaluation report.
    """
    print("=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=["No Churn", "Churn"]))

    print("\nCONFUSION MATRIX")
    print("-" * 50)
    print(compute_confusion_matrix(y_true, y_pred))

    if y_proba is not None:
        metrics = compute_classification_metrics(y_true, y_pred, y_proba)
        print(f"\nROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"PR-AUC:  {metrics['pr_auc']:.4f}")

        print("\nLIFT TABLE (Top Deciles)")
        print("-" * 50)
        lift_df = compute_lift_table(y_true, y_proba)
        print(
            lift_df[["decile", "n_churners", "churn_rate", "lift", "cum_churn_captured"]]
            .head(5)
            .to_string(index=False)
        )

        top_20_capture = lift_df[lift_df["cum_customers_pct"] <= 0.2]["cum_churn_captured"].max()
        print(f"\n→ Top 20% captures {top_20_capture:.1%} of all churners")
