"""
evaluate_business_metrics.py – "Đánh giá mô hình theo ngôn ngữ kinh doanh"
(Propensity Model methodology, §6-7): decile/lift, KS, precision@20%/
capture@20% (the exact business cut this project targets — top 20%
priority customers), a calibration curve + Brier score, and a SHAP
summary plot.

This is deliberately separate from evaluate_model.py, which stays the
lightweight AUC-only check the Airflow DAG can run cheaply and often.
This script is the deeper, "read before you promote" report — run it
whenever a new candidate needs to be judged against the actual campaign
use case (rank the top slice), not just discrimination on the whole
population.

Business metrics are also written back onto the training run of the
evaluated model version (via MlflowClient.log_metric with the original
run_id), so promote_model.py can read capture_at_20pct for both
"candidate" and "prod" without re-running this script inline.
"""

import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from mlflow import MlflowClient
from scipy.stats import ks_2samp
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.feature_engineering import (
    FEATURE_COLUMNS,
    build_churn_labels,
    compute_features_for_all_customers,
)
from src.logging_utils import setup_logging

logger = setup_logging("evaluate_business_metrics")

MODEL_NAME           = "Churn_model"
MLFLOW_TRACKING_URI  = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
INACTIVITY_WINDOW    = int(os.environ.get("INACTIVITY_WINDOW", "30"))
TOP_K_PCT            = 0.20  # the business goal: top-20% priority customers
REPORTS_DIR          = os.environ.get("REPORTS_DIR", "reports/business_eval")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_evaluation_data(transactions_file: str) -> pd.DataFrame:
    """Same feature+label construction as evaluate_model.py, for comparability."""
    transactions = pd.read_csv(transactions_file)
    transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])

    # Same fix as evaluate_model.py: reserve INACTIVITY_WINDOW trailing days
    # as the forward-looking window, otherwise every customer is trivially
    # "churned" (no days left to observe a future purchase).
    snapshot_date = transactions["transaction_date"].max() - pd.Timedelta(days=INACTIVITY_WINDOW)
    features = compute_features_for_all_customers(transactions, snapshot_date)
    labels = build_churn_labels(transactions, snapshot_date, INACTIVITY_WINDOW)
    return features.merge(labels, on="customer_id")


# ---------------------------------------------------------------------------
# Business metrics
# ---------------------------------------------------------------------------

def decile_lift_table(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    """Decile 1 = top 10% of scores. Lift and cumulative capture per decile,
    per the Propensity Model methodology §6.1."""
    df = pd.DataFrame({"y": y_true, "score": y_score})
    df["decile"] = 10 - pd.qcut(df["score"], 10, labels=False, duplicates="drop")
    overall_rate = df["y"].mean()

    table = (
        df.groupby("decile")
        .agg(n_customers=("y", "size"), n_positive=("y", "sum"))
        .sort_index()
    )
    table["conversion_rate"] = table["n_positive"] / table["n_customers"]
    table["lift"] = table["conversion_rate"] / overall_rate
    table["cumulative_capture_rate"] = table["n_positive"].cumsum() / table["n_positive"].sum()
    return table.reset_index()


def precision_capture_at_k(y_true: np.ndarray, y_score: np.ndarray, k_pct: float) -> dict:
    df = pd.DataFrame({"y": y_true, "score": y_score}).sort_values("score", ascending=False)
    k = max(1, int(len(df) * k_pct))
    top_k = df.head(k)
    return {
        f"precision_at_{int(k_pct*100)}pct": float(top_k["y"].mean()),
        f"capture_at_{int(k_pct*100)}pct":   float(top_k["y"].sum() / df["y"].sum()),
        f"n_targeted_at_{int(k_pct*100)}pct": k,
    }


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(ks_2samp(y_score[y_true == 1], y_score[y_true == 0]).statistic)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(transactions_file: str, model_alias: str = "candidate") -> dict:
    client = MlflowClient()

    logger.info("Building evaluation dataset …")
    data = load_evaluation_data(transactions_file)
    X = data[FEATURE_COLUMNS]
    y = data["churn"].to_numpy()

    model_uri = f"models:/{MODEL_NAME}@{model_alias}"
    mv = client.get_model_version_by_alias(MODEL_NAME, model_alias)
    logger.info("Loading model '%s' (version %s, run %s) …", model_uri, mv.version, mv.run_id)
    model = mlflow.sklearn.load_model(model_uri)

    proba = model.predict_proba(X)[:, 1]

    decile_table = decile_lift_table(y, proba)
    top_k = precision_capture_at_k(y, proba, TOP_K_PCT)
    ks = ks_statistic(y, proba)
    brier = brier_score_loss(y, proba)
    prob_true, prob_pred = calibration_curve(y, proba, n_bins=10)

    metrics = {
        "ks_statistic": ks,
        "brier_score": brier,
        "decile1_lift": float(decile_table.loc[decile_table["decile"] == 1, "lift"].iloc[0]),
        **top_k,
    }
    logger.info("Business metrics: %s", metrics)
    logger.info("Decile/lift table:\n%s", decile_table.to_string(index=False))

    # ---- Artifacts -------------------------------------------------------
    run_dir = os.path.join(
        REPORTS_DIR, f"{model_alias}_v{mv.version}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    os.makedirs(run_dir, exist_ok=True)

    decile_path = os.path.join(run_dir, "decile_lift_table.csv")
    decile_table.to_csv(decile_path, index=False)

    calib_path = os.path.join(run_dir, "calibration_curve.png")
    plt.figure(figsize=(5, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed churn rate")
    plt.title(f"Calibration curve — Brier={brier:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(calib_path, dpi=120)
    plt.close()

    lift_path = os.path.join(run_dir, "decile_lift_chart.png")
    plt.figure(figsize=(6, 4))
    plt.bar(decile_table["decile"], decile_table["lift"])
    plt.axhline(1.0, color="gray", linestyle="--")
    plt.xlabel("Decile (1 = highest predicted risk)")
    plt.ylabel("Lift vs. overall churn rate")
    plt.title("Decile / lift analysis")
    plt.tight_layout()
    plt.savefig(lift_path, dpi=120)
    plt.close()

    shap_path = os.path.join(run_dir, "shap_summary.png")
    try:
        explainer = shap.TreeExplainer(model.named_steps["lgbm"])
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        plt.savefig(shap_path, dpi=120)
        plt.close()
    except Exception:
        logger.exception("SHAP summary plot failed — continuing without it.")
        shap_path = None

    # ---- Log to a dedicated business-eval experiment ----------------------
    mlflow.set_experiment(f"{MODEL_NAME}_BusinessEval")
    with mlflow.start_run(run_name=f"business_eval_{model_alias}_v{mv.version}") as run:
        mlflow.log_params({
            "model_alias": model_alias, "model_version": mv.version,
            "model_uri": model_uri, "top_k_pct": TOP_K_PCT,
        })
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(decile_path)
        mlflow.log_artifact(calib_path)
        mlflow.log_artifact(lift_path)
        if shap_path:
            mlflow.log_artifact(shap_path)
        logger.info("Business-eval run logged: %s", run.info.run_id)

    # ---- Also attach the headline business metric to the model version's
    # own training run, so promote_model.py can read it directly. ----------
    try:
        client.log_metric(mv.run_id, "capture_at_20pct", metrics["capture_at_20pct"])
        client.log_metric(mv.run_id, "precision_at_20pct", metrics["precision_at_20pct"])
        logger.info("Attached capture_at_20pct=%.4f to training run %s.",
                    metrics["capture_at_20pct"], mv.run_id)
    except Exception:
        logger.exception("Could not attach business metric to training run %s.", mv.run_id)

    logger.info("Artifacts written to %s", run_dir)
    return metrics


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Usage: python evaluate_business_metrics.py <transactions_csv> [alias]")
        sys.exit(1)

    alias = sys.argv[2] if len(sys.argv) > 2 else "candidate"
    evaluate(sys.argv[1], alias)
    sys.exit(0)
