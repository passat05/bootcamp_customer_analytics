"""
build_features.py – Materializes analytics-ready training datasets from raw
customer/transaction data.

This is the "transform" stage between data ingestion and model training:
event-level CSVs (customers.csv, transactions.csv) are aggregated into the
per-customer feature/label tables that train_model.py and
train_survival_model.py consume via --dataset. Runs independently of
training so the same snapshot can be inspected or reused by both models,
and so the Airflow DAG graph reflects a real transform -> train separation
instead of every training run re-deriving features from raw data.

Uses src/feature_engineering.py's build_churn_training_dataset() and
build_survival_training_dataset() — the same functions available to any
other consumer, so this script and a from-scratch training run always
agree on what "the dataset" is.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.feature_engineering import (
    build_churn_training_dataset,
    build_survival_training_dataset,
)
from src.logging_utils import setup_logging

logger = setup_logging("build_features")

INACTIVITY_WINDOW = int(os.environ.get("INACTIVITY_WINDOW", "30"))


def build(customers_file: str, transactions_file: str, output_dir: str) -> None:
    customers = pd.read_csv(customers_file)
    customers["signup_date"] = pd.to_datetime(customers["signup_date"])

    transactions = pd.read_csv(transactions_file)
    transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])

    churn_dataset = build_churn_training_dataset(transactions, INACTIVITY_WINDOW)
    survival_dataset = build_survival_training_dataset(customers, transactions, INACTIVITY_WINDOW)

    os.makedirs(output_dir, exist_ok=True)
    churn_path = os.path.join(output_dir, "churn_dataset.csv")
    survival_path = os.path.join(output_dir, "survival_dataset.csv")

    churn_dataset.to_csv(churn_path, index=False)
    survival_dataset.to_csv(survival_path, index=False)

    logger.info(
        "churn_dataset: %d rows, churn rate %.1f%% -> %s",
        len(churn_dataset), churn_dataset["churn"].mean() * 100, churn_path,
    )
    logger.info(
        "survival_dataset: %d rows, event rate %.1f%% -> %s",
        len(survival_dataset), survival_dataset["event"].mean() * 100, survival_path,
    )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        logger.error("Usage: python build_features.py <customers_csv> <transactions_csv> <output_dir>")
        sys.exit(1)

    build(sys.argv[1], sys.argv[2], sys.argv[3])
    sys.exit(0)
