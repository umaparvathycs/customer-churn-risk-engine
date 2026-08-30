"""Train, evaluate, and persist the churn model."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from data_generation import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET, generate_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "customer_churn.csv"
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"


def train_model() -> dict:
    if not DATA_PATH.exists():
        generate_dataset().to_csv(DATA_PATH, index=False)
    df = pd.read_csv(DATA_PATH)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, stratify=y, random_state=42)

    preprocessor = ColumnTransformer([
        ("numeric", "passthrough", NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])
    model = RandomForestClassifier(
        n_estimators=350, max_depth=12, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1
    )
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    model.fit(X_train_t, y_train)
    probabilities = model.predict_proba(X_test_t)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "classification_report": classification_report(y_test, predictions, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "test_rows": int(len(y_test)), "churn_rate": round(float(y.mean()), 4),
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"preprocessor": preprocessor, "model": model, "features": FEATURES, "numeric_features": NUMERIC_FEATURES, "categorical_features": CATEGORICAL_FEATURES}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({"roc_auc": metrics["roc_auc"], "test_rows": metrics["test_rows"]}, indent=2))
    return metrics


if __name__ == "__main__":
    train_model()
