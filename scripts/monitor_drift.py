"""
monitor_drift.py – Basic MLOps monitoring: Population Stability Index (PSI)
between the training feature distribution and the current scoring
population, per the Propensity Model methodology §9.1.

Deliberately no Grafana/Prometheus wiring — this is the "cheap check that
catches 80% of pipeline incidents" the methodology describes: a script you
can run ad hoc or from an Airflow task, producing a small CSV report.

PSI thresholds (standard convention, also used in the methodology):
  < 0.10            stable
  0.10 - 0.25       watch
  > 0.25            significant shift — investigate before trusting the model
"""

import os
import sys
from datetime import datetime

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.feature_engineering import FEATURE_COLUMNS, compute_features_for_all_customers
from src.logging_utils import setup_logging

logger = setup_logging("monitor_drift")

MODEL_NAME          = "Churn_model"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
REPORTS_DIR         = os.environ.get("REPORTS_DIR", "reports/drift")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """
    PSI = sum_b (a_b - e_b) * ln(a_b / e_b)
    a_b, e_b = share of observations in bin b, current ("actual") vs.
    reference ("expected"/training). Bin edges are reference quantiles, so
    the reference distribution is uniform across bins by construction.
    """
    reference = reference.dropna()
    current = current.dropna()

    breakpoints = np.unique(reference.quantile(np.linspace(0, 1, bins + 1)).to_numpy())
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    if len(breakpoints) < 3:
        # Degenerate reference distribution (e.g. a near-constant feature) —
        # PSI isn't meaningful with fewer than 2 real bins.
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    eps = 1e-6
    ref_pct = np.clip(ref_counts / max(len(reference), 1), eps, None)
    cur_pct = np.clip(cur_counts / max(len(current), 1), eps, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _status(psi: float) -> str:
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "watch"
    return "shift"


def run(reference_dataset_csv: str, transactions_csv: str, model_alias: str | None = None) -> pd.DataFrame:
    reference = pd.read_csv(reference_dataset_csv)
    missing = [c for c in FEATURE_COLUMNS if c not in reference.columns]
    if missing:
        raise ValueError(f"Reference dataset is missing feature columns: {missing}")

    transactions = pd.read_csv(transactions_csv)
    transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])
    snapshot_date = transactions["transaction_date"].max()
    current = compute_features_for_all_customers(transactions, snapshot_date)

    rows = []
    for col in FEATURE_COLUMNS:
        psi = population_stability_index(reference[col], current[col])
        rows.append({"feature": col, "psi": psi, "status": _status(psi)})

    if model_alias:
        try:
            model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{model_alias}")
            ref_scores = pd.Series(model.predict_proba(reference[FEATURE_COLUMNS])[:, 1])
            cur_scores = pd.Series(model.predict_proba(current[FEATURE_COLUMNS])[:, 1])
            psi = population_stability_index(ref_scores, cur_scores)
            rows.append({"feature": f"model_score[{model_alias}]", "psi": psi, "status": _status(psi)})
        except Exception:
            logger.exception("Could not score model alias '%s' for score-level PSI — feature PSI still computed.", model_alias)

    report = pd.DataFrame(rows)
    logger.info("Drift report (snapshot_date=%s):\n%s", snapshot_date.date(), report.to_string(index=False))

    shifted = report[report["status"] == "shift"]
    if not shifted.empty:
        logger.warning("Significant drift detected on: %s", ", ".join(shifted["feature"]))
    watch = report[report["status"] == "watch"]
    if not watch.empty:
        logger.info("Features to watch: %s", ", ".join(watch["feature"]))

    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, f"drift_report_{datetime.now():%Y%m%d_%H%M%S}.csv")
    report.to_csv(out_path, index=False)
    logger.info("Report written to %s", out_path)

    return report


if __name__ == "__main__":
    if len(sys.argv) < 3:
        logger.error(
            "Usage: python monitor_drift.py <reference_dataset_csv> <transactions_csv> [model_alias]"
        )
        sys.exit(1)

    alias = sys.argv[3] if len(sys.argv) > 3 else None
    run(sys.argv[1], sys.argv[2], alias)
    sys.exit(0)
