"""
evaluate_model.py – Production-ready model evaluation script.

Imports feature computation from src/feature_engineering.py —
the same source of truth used by train_model.py and ChurnPredictor.
This guarantees evaluation uses identical features to training and serving.
"""

import os
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ✅ Same source of truth as train_model.py and ChurnPredictor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.feature_engineering import (
    FEATURE_COLUMNS,
    build_churn_labels,
    compute_features_for_all_customers,
)
from src.logging_utils import setup_logging

logger = setup_logging("evaluate_model")

MODEL_NAME          = "Churn_model"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
INACTIVITY_WINDOW   = int(os.environ.get("INACTIVITY_WINDOW", "30"))

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def load_evaluation_data(customers_file: str, transactions_file: str) -> pd.DataFrame:
    """Load CSVs and build the same feature+label dataset as training."""
    transactions = pd.read_csv(transactions_file)
    transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])

    # Reserve the trailing INACTIVITY_WINDOW days as the forward-looking
    # window used to observe the label — snapshotting at max() itself would
    # leave zero days to observe future activity, so every customer would be
    # labeled "churned" by construction (a self-fulfilling label, not a real
    # measurement).
    snapshot_date = transactions["transaction_date"].max() - pd.Timedelta(days=INACTIVITY_WINDOW)

    features = compute_features_for_all_customers(transactions, snapshot_date)
    labels   = build_churn_labels(transactions, snapshot_date, INACTIVITY_WINDOW)

    return features.merge(labels, on="customer_id")


def evaluate_model(
    customers_file: str,
    transactions_file: str,
    model_alias: str = "prod",
) -> dict:
    """
    Evaluate the registered MLflow model and log results to a new run.

    Parameters
    ----------
    customers_file, transactions_file : str
        Paths to CSV files.
    model_alias : str
        MLflow model alias to evaluate (default: 'prod').

    Returns
    -------
    dict of metric name → value
    """
    logger.info("Building evaluation dataset …")
    data = load_evaluation_data(customers_file, transactions_file)

    X = data[FEATURE_COLUMNS]   # ✅ Same column list as training
    y = data["churn"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model_uri = f"models:/{MODEL_NAME}@{model_alias}"
    logger.info("Loading model from '%s' …", model_uri)
    model = mlflow.sklearn.load_model(model_uri)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy":      accuracy_score(y_test, preds),
        "roc_auc":       roc_auc_score(y_test, proba),
        "avg_precision": average_precision_score(y_test, proba),
    }

    mlflow.set_experiment(f"{MODEL_NAME}_Evaluation")
    with mlflow.start_run(run_name=f"eval_{model_alias}") as run:
        mlflow.log_params({"model_alias": model_alias, "model_uri": model_uri})
        mlflow.log_metrics(metrics)
        logger.info("Metrics: %s  (Run ID: %s)", metrics, run.info.run_id)

    return metrics


if __name__ == "__main__":
    if len(sys.argv) < 3:
        logger.error(
            "Usage: python evaluate_model.py <customers_csv> <transactions_csv> [alias]"
        )
        sys.exit(1)

    alias   = sys.argv[3] if len(sys.argv) > 3 else "prod"
    results = evaluate_model(sys.argv[1], sys.argv[2], alias)
    logger.info("Done. ROC-AUC = %.4f", results["roc_auc"])
    sys.exit(0)