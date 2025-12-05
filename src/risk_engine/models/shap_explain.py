from pathlib import Path
import joblib
import shap
import numpy as np
import pandas as pd

from risk_engine.data.preprocess import split_features_target
from risk_engine.data.feature_engineering import apply_feature_engineering

df_raw, DATA_DIR = load_raw()
MODELS_DIR = DATA_DIR.parent / "models"


class ShapExplainerService:
    def __init__(self):
        self.clf = joblib.load(MODELS_DIR / "model.pkl")
        self.preprocessor = self.clf.named_steps["preprocessor"]
        self.model = self.clf.named_steps["model"]
        self.explainer = shap.TreeExplainer(self.model)

    def explain_single(self, X: pd.DataFrame):
        X_fe = apply_feature_engineering(X)
        X_proc = self.preprocessor.transform(X_fe)
        shap_values = self.explainer(X_proc)
        return shap_values
