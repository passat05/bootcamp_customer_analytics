"""
train_survival_model.py – MLflow training script for the CoxPH survival model.

Covariates (frequency, avg_amount) are computed from each customer's FULL
transaction history to match SurvivalAnalyzer._extract_features() at serving
time (src/survival_analyzer.py) — this avoids training-serving skew.

Duration/event reuse the same snapshot_date + inactivity-window definition as
train_model.py's churn label (via src/feature_engineering.py), so "event=1"
here means the same thing as "churn=1" there.

Every trained version is tagged with the "candidate" alias, never "prod"
directly — scripts/promote_model.py is the only thing that moves "prod",
and only promotes a candidate that beats the currently served model.
"""

import argparse
import os
import sys

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from mlflow import MlflowClient
from sksurv.metrics import integrated_brier_score
from sksurv.util import Surv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.feature_engineering import build_survival_training_dataset
from src.logging_utils import setup_logging

logger = setup_logging("train_survival_model")

MODEL_NAME          = "Survival_model"
MLFLOW_TRACKING_URI  = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
INACTIVITY_WINDOW    = int(os.environ.get("INACTIVITY_WINDOW", "30"))

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(f"{MODEL_NAME}_Experiment")


def load_training_data(
    customers_file: str | None = None,
    transactions_file: str | None = None,
    dataset_file: str | None = None,
) -> pd.DataFrame:
    """Load a pre-built (frequency, avg_amount, duration, event) dataset, or build one from raw CSVs."""
    if dataset_file:
        data = pd.read_csv(dataset_file)
        logger.info("Loaded pre-built dataset: %s (%d rows)", dataset_file, len(data))
        return data

    customers = pd.read_csv(customers_file)
    customers["signup_date"] = pd.to_datetime(customers["signup_date"])

    transactions = pd.read_csv(transactions_file)
    transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])

    data = build_survival_training_dataset(customers, transactions, INACTIVITY_WINDOW)
    logger.info(
        "Built dataset from raw data: %d samples — event rate %.1f%%",
        len(data), data["event"].mean() * 100,
    )
    return data


def compute_integrated_brier_score(cph: CoxPHFitter, data: pd.DataFrame) -> float:
    """
    In-sample IBS over the observed duration range (same recipe as the
    exploratory notebook: customer_analysis.ipynb, cell using sksurv).
    """
    surv_funcs = cph.predict_survival_function(data[["frequency", "avg_amount"]])

    t_min = data.loc[data["event"] == 1, "duration"].min()
    t_max = data["duration"].max()
    times = np.linspace(t_min, t_max * 0.999, 100)

    surv_preds = np.asarray([
        np.interp(times, surv_funcs.index.values, surv_funcs.iloc[:, i].values)
        for i in range(surv_funcs.shape[1])
    ])

    y = Surv.from_arrays(event=data["event"].astype(bool), time=data["duration"])
    return float(integrated_brier_score(y, y, surv_preds, times))


def train(
    customers_file: str | None = None,
    transactions_file: str | None = None,
    dataset_file: str | None = None,
) -> dict:
    data = load_training_data(customers_file, transactions_file, dataset_file)

    cph = CoxPHFitter()

    with mlflow.start_run() as run:
        mlflow.set_tags({
            "model_type":             "CoxPHFitter",
            "feature_set":            "frequency_avg_amount_v1",
            "inactivity_window_days": str(INACTIVITY_WINDOW),
        })

        cph.fit(
            data[["frequency", "avg_amount", "duration", "event"]],
            duration_col="duration",
            event_col="event",
        )

        metrics = {
            "concordance_index":     cph.concordance_index_,
            "integrated_brier_score": compute_integrated_brier_score(cph, data),
        }
        mlflow.log_params({"inactivity_window": INACTIVITY_WINDOW})
        mlflow.log_metrics(metrics)
        logger.info("Metrics: %s", metrics)

        mlflow.sklearn.log_model(
            sk_model=cph,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            input_example=data[["frequency", "avg_amount"]].head(1),
        )

        client   = MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=None)
        if not versions:
            raise RuntimeError(f"No versions found for model '{MODEL_NAME}' after logging.")

        latest        = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
        model_version = latest.version

        client.set_registered_model_alias(
            name=MODEL_NAME, version=model_version, alias="candidate"
        )
        logger.info(
            "Model '%s' v%s tagged as 'candidate'. Run ID: %s. "
            "Run promote_model.py to compare against 'prod' and promote if better.",
            MODEL_NAME, model_version, run.info.run_id,
        )

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("customers_csv", nargs="?", help="Path to customers.csv (raw mode)")
    parser.add_argument("transactions_csv", nargs="?", help="Path to transactions.csv (raw mode)")
    parser.add_argument("--dataset", help="Path to a pre-built dataset from build_features.py")
    args = parser.parse_args()

    if args.dataset:
        result = train(dataset_file=args.dataset)
    elif args.customers_csv and args.transactions_csv:
        result = train(customers_file=args.customers_csv, transactions_file=args.transactions_csv)
    else:
        parser.error("Provide either --dataset <path>, or <customers_csv> <transactions_csv>")

    logger.info(
        "Training finished. Concordance index = %.4f, Integrated Brier Score = %.4f",
        result["concordance_index"], result["integrated_brier_score"],
    )
    sys.exit(0)
