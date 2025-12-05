# src/models/train_model.py
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

from src.data.preprocess import load_raw, split_features_target, build_preprocessor
from src.data.feature_engineering import apply_feature_engineering

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "src" / "models"

_model_cache = None

def get_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = joblib.load(MODELS_DIR / "model.pkl")
    return _model_cache

def train_and_save_model():
    df, _ = load_raw()
    df = apply_feature_engineering(df)

    X, y, _ = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    clf = Pipeline([("preprocessor", preprocessor), ("model", xgb)])
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy: {acc:.3f}")
    print(f"Recall:   {rec:.3f}")
    print(f"AUC:      {auc:.3f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "model.pkl"
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train_and_save_model()
