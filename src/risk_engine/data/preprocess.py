import os
from pathlib import Path
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from risk_engine.data.feature_engineering import apply_feature_engineering

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

TARGET_COL = "Churn"


def load_raw() -> pd.DataFrame:

    # Download latest version
    path = Path(kagglehub.dataset_download("blastchar/telco-customer-churn"))

    #print("Path to dataset files:", path)

    data_path = os.path.join(path, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

    df = pd.read_csv(data_path)


    df.columns = df.columns.str.strip()

    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].str.strip()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "tenure" in df.columns:
        df["tenure"] = df["tenure"].astype("Int64")

    return df, path


def split_features_target(df: pd.DataFrame):
    id_cols = [c for c in df.columns if "customer" in c.lower() or "id" in c.lower()]
    X = df.drop(columns=id_cols + [TARGET_COL])
    y = df[TARGET_COL].map({"No": 0, "Yes": 1})
    return X, y, id_cols


def get_feature_types(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols, categorical_cols = get_feature_types(X)

    numeric_transformer = Pipeline(steps=[
                            ("imputer", SimpleImputer(strategy="median")),          # of "mean"
                            ("scaler", StandardScaler())
                        ])
    categorical_transformer = Pipeline(steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),   # vult NaN met modus
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))
                        ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor

def build_dataset():
    df, _ = load_raw()
    df = apply_feature_engineering(df)
    X, y, id_cols = split_features_target(df)
    return df, X, y, id_cols
