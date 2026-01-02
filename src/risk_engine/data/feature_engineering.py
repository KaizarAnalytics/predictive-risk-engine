import pandas as pd


def add_tenure_buckets(df: pd.DataFrame) -> pd.DataFrame:
    if "tenure" not in df.columns:
        return df

    # Ensure tenure is numeric
    if not pd.api.types.is_numeric_dtype(df["tenure"]):
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")

    df = df.copy()
    df["tenure_bucket"] = pd.cut(
        df["tenure"].fillna(0),
        bins=[-1, 12, 24, 48, 72, 1000],
        labels=["0-12", "13-24", "25-48", "49-72", "72+"]
    )
    return df


def add_charge_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "TotalCharges" in df.columns and "tenure" in df.columns:
        df["avg_monthly_total"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    return df


def add_binary_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Example flags - adjust as needed
    if "Contract" in df.columns:
        df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    if "InternetService" in df.columns:
        df["has_fiber"] = (df["InternetService"] == "Fiber optic").astype(int)
    if "PhoneService" in df.columns and "InternetService" in df.columns:
        df["multi_service"] = (
            (df["PhoneService"] == "Yes") & (df["InternetService"] != "No")
        ).astype(int)
    return df


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure numeric columns are properly typed
    numeric_cols = ["TotalCharges", "MonthlyCharges", "tenure", "SeniorCitizen"]
    for col in numeric_cols:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = add_tenure_buckets(df)
    df = add_charge_ratios(df)
    df = add_binary_flags(df)
    return df
