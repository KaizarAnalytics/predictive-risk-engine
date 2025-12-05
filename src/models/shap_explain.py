# src/models/shap_explain.py
from pathlib import Path
import joblib
import shap
import numpy as np
import pandas as pd

from src.data.preprocess import split_features_target
from src.data.feature_engineering import apply_feature_engineering

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "src" / "models"


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
