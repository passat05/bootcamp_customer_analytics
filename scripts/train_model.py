"""
train_model.py – Production-ready MLflow training script.

Feature computation is delegated to src/feature_engineering.py —
the same module used by ChurnPredictor at serving time.
This eliminates training-serving skew entirely.

Every trained version is tagged with the "candidate" alias, never "prod"
directly — scripts/promote_model.py is the only thing that moves "prod",
and only promotes a candidate that beats the currently served model.
"""

import argparse
import os
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline

# ✅ Shared feature engineering — same as ChurnPredictor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.feature_engineering import (
    FEATURE_COLUMNS,
    build_churn_training_dataset,
    build_churn_training_dataset_stacked,
)
from src.logging_utils import setup_logging

logger = setup_logging("train_model")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME           = "Churn_model"
MLFLOW_TRACKING_URI  = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
INACTIVITY_WINDOW    = int(os.environ.get("INACTIVITY_WINDOW", "30"))

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(f"{MODEL_NAME}_Experiment")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data(
    customers_file: str | None = None,
    transactions_file: str | None = None,
    dataset_file: str | None = None,
) -> pd.DataFrame:
    """Load a pre-built dataset (from build_features.py), or build one from raw CSVs."""
    if dataset_file:
        data = pd.read_csv(dataset_file)
        logger.info("Loaded pre-built dataset: %s (%d rows)", dataset_file, len(data))
        return data

    transactions = pd.read_csv(transactions_file)
    transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])

    data = build_churn_training_dataset(transactions, INACTIVITY_WINDOW)
    logger.info(
        "Built dataset from raw data: %d samples — churn rate %.1f%%",
        len(data), data["churn"].mean() * 100,
    )
    return data


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    customers_file: str | None = None,
    transactions_file: str | None = None,
    dataset_file: str | None = None,
) -> float:
    data = load_training_data(customers_file, transactions_file, dataset_file)

    X = data[FEATURE_COLUMNS]   # ✅ Same column list as ChurnPredictor
    y = data["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    logger.info('Class distribution — neg: %d, pos: %d', neg, pos)
    if pos == 0 or neg == 0:
        raise ValueError(
            f'Training data has only one class (neg={neg}, pos={pos}). '
            'Check snapshot_date and INACTIVITY_WINDOW settings.'
        )
    scale_pos_weight = neg / pos

    params = {
        "n_estimators":     300,
        "learning_rate":    0.05,
        "max_depth":        3,
        "random_state":     42,
        "n_jobs":           -1,
        "scale_pos_weight": scale_pos_weight,
        "verbosity":        -1,
    }

    # LightGBM is a tree model — no StandardScaler needed
    pipeline = Pipeline([("lgbm", LGBMClassifier(**params))])

    with mlflow.start_run() as run:
        mlflow.set_tags({
            "model_type":            "LGBMClassifier",
            "feature_set":           "RFM_v1",
            "inactivity_window_days": str(INACTIVITY_WINDOW),
        })

        pipeline.fit(X_train, y_train)

        proba_test = pipeline.predict_proba(X_test)[:, 1]
        preds_test = (proba_test >= 0.5).astype(int)

        metrics = {
            "accuracy":         accuracy_score(y_test, preds_test),
            "roc_auc":          roc_auc_score(y_test, proba_test),
            "avg_precision":    average_precision_score(y_test, proba_test),
            "churn_rate_train": float(y_train.mean()),
            "churn_rate_test":  float(y_test.mean()),
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        logger.info("Metrics: %s", metrics)

        signature = mlflow.models.infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            input_example=X_train.head(1),
            signature=signature,
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

    return metrics["roc_auc"]


# ---------------------------------------------------------------------------
# Stacked-cohort training with out-of-time (OOT) validation
# ---------------------------------------------------------------------------

def train_stacked(
    transactions_file: str,
    snapshot_freq_days: int = 30,
    min_history_days: int = 90,
) -> dict:
    """
    Train on stacked monthly cohorts (see the Propensity Model methodology,
    §2.2/§4.1) instead of a single 80%-of-timeline snapshot.

    Validation follows the two-layer scheme the methodology recommends:
      - in-time: a held-out slice of CUSTOMERS from the same cohorts used
        for training (catches ordinary overfitting)
      - out-of-time (OOT): the single most-recent cohort, entirely unseen
        during training (the honest estimate of future performance — the
        gap between in-time and OOT is reported explicitly rather than
        hidden behind one AUC number)

    Splitting is done by customer_id (GroupShuffleSplit), never by row —
    stacked cohorts put the same customer in multiple rows, and a row-level
    split would leak a customer's other-snapshot label into the "held out"
    set.
    """
    transactions = pd.read_csv(transactions_file)
    transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])

    stacked = build_churn_training_dataset_stacked(
        transactions,
        INACTIVITY_WINDOW,
        snapshot_freq_days=snapshot_freq_days,
        min_history_days=min_history_days,
    )

    oot_snapshot = stacked["snapshot_date"].max()
    oot = stacked[stacked["snapshot_date"] == oot_snapshot]
    pool = stacked[stacked["snapshot_date"] != oot_snapshot]

    if pool.empty:
        raise ValueError(
            "Stacked cohorts produced only one snapshot — nothing left to train on "
            "after reserving the OOT holdout. Reduce snapshot_freq_days or provide more data."
        )

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, intime_idx = next(gss.split(pool, groups=pool["customer_id"]))
    train_df = pool.iloc[train_idx]
    intime_df = pool.iloc[intime_idx]

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["churn"]
    X_intime, y_intime = intime_df[FEATURE_COLUMNS], intime_df["churn"]
    X_oot, y_oot = oot[FEATURE_COLUMNS], oot["churn"]

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    logger.info("Class distribution (train) — neg: %d, pos: %d", neg, pos)
    if pos == 0 or neg == 0:
        raise ValueError(
            f"Stacked training data has only one class (neg={neg}, pos={pos}). "
            "Check snapshot_freq_days / min_history_days / INACTIVITY_WINDOW settings."
        )
    scale_pos_weight = neg / pos

    params = {
        "n_estimators":     300,
        "learning_rate":    0.05,
        "max_depth":        3,
        "random_state":     42,
        "n_jobs":           -1,
        "scale_pos_weight": scale_pos_weight,
        "verbosity":        -1,
    }
    pipeline = Pipeline([("lgbm", LGBMClassifier(**params))])

    with mlflow.start_run() as run:
        mlflow.set_tags({
            "model_type":            "LGBMClassifier",
            "feature_set":           "RFM_v1",
            "inactivity_window_days": str(INACTIVITY_WINDOW),
            "training_mode":         "stacked_cohorts",
            "n_snapshots":           str(stacked["snapshot_date"].nunique()),
            "oot_snapshot_date":     str(oot_snapshot.date()),
        })

        pipeline.fit(X_train, y_train)

        def _score(X, y, prefix: str) -> dict:
            proba = pipeline.predict_proba(X)[:, 1]
            preds = (proba >= 0.5).astype(int)
            return {
                f"{prefix}_accuracy":      accuracy_score(y, preds),
                f"{prefix}_roc_auc":       roc_auc_score(y, proba),
                f"{prefix}_avg_precision": average_precision_score(y, proba),
                f"{prefix}_churn_rate":    float(y.mean()),
            }

        metrics = {}
        metrics.update(_score(X_intime, y_intime, "intime"))
        metrics.update(_score(X_oot, y_oot, "oot"))
        metrics["churn_rate_train"] = float(y_train.mean())
        # Primary metric kept at the same key promote_model.py already reads,
        # sourced from the OOT split — the honest, decision-relevant number.
        metrics["roc_auc"] = metrics["oot_roc_auc"]
        metrics["avg_precision"] = metrics["oot_avg_precision"]
        metrics["accuracy"] = metrics["oot_accuracy"]

        oot_gap = metrics["intime_roc_auc"] - metrics["oot_roc_auc"]
        metrics["intime_vs_oot_auc_gap"] = oot_gap

        mlflow.log_params(params)
        mlflow.log_params({
            "snapshot_freq_days": snapshot_freq_days,
            "min_history_days": min_history_days,
        })
        mlflow.log_metrics(metrics)
        logger.info("In-time vs OOT AUC gap: %.4f (in-time=%.4f, OOT=%.4f)",
                    oot_gap, metrics["intime_roc_auc"], metrics["oot_roc_auc"])
        logger.info("Metrics: %s", metrics)

        signature = mlflow.models.infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            input_example=X_train.head(1),
            signature=signature,
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
            "Model '%s' v%s (stacked-cohort) tagged as 'candidate'. Run ID: %s.",
            MODEL_NAME, model_version, run.info.run_id,
        )

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("customers_csv", nargs="?", help="Path to customers.csv (raw mode)")
    parser.add_argument("transactions_csv", nargs="?", help="Path to transactions.csv (raw mode)")
    parser.add_argument("--dataset", help="Path to a pre-built dataset from build_features.py")
    parser.add_argument(
        "--stacked", action="store_true",
        help="Train on stacked monthly cohorts with in-time/OOT validation "
             "(raw-CSV mode only — needs transactions_csv, not --dataset)",
    )
    args = parser.parse_args()

    if args.stacked:
        if not args.transactions_csv:
            parser.error("--stacked requires <transactions_csv> (raw-CSV mode)")
        metrics = train_stacked(args.transactions_csv)
        logger.info(
            "Training finished (stacked). OOT ROC-AUC = %.4f, in-time ROC-AUC = %.4f, gap = %.4f",
            metrics["oot_roc_auc"], metrics["intime_roc_auc"], metrics["intime_vs_oot_auc_gap"],
        )
        sys.exit(0)

    if args.dataset:
        auc = train(dataset_file=args.dataset)
    elif args.customers_csv and args.transactions_csv:
        auc = train(customers_file=args.customers_csv, transactions_file=args.transactions_csv)
    else:
        parser.error("Provide either --dataset <path>, or <customers_csv> <transactions_csv>, or --stacked")

    logger.info("Training finished. ROC-AUC = %.4f", auc)
    sys.exit(0)
