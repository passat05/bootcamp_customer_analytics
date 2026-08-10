"""
promote_model.py – The only script allowed to move the "prod" MLflow alias.

train_model.py and train_survival_model.py tag every freshly logged version
as "candidate", never "prod" directly. This script only promotes a candidate
that beats the currently served "prod" version, or if no "prod" exists yet.

For Churn_model, when a transactions CSV is provided, promotion is decided
by an apples-to-apples comparison: both candidate and prod are re-scored on
ONE identical, freshly-built holdout, instead of trusting each version's own
historically logged metric. This is not a theoretical concern — it is the
fix for a real incident: an early version (trained before a point-in-time
feature-leakage bug was fixed) got stuck as "prod" with an inflated
self-reported AUC of 0.98 that no honestly-trained candidate could ever beat
by comparing against its own logged number, because the leak was baked into
that number, not into anything a fresh comparison would reproduce. See
"Known issue" in docs/model_card_churn.md for the full writeup.

Without a transactions CSV (or for Survival_model, which this fresh-holdout
path does not yet cover), promotion falls back to comparing each alias's own
logged metric — weaker, but still better than no gate at all.
"""

import os
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.feature_engineering import (
    FEATURE_COLUMNS,
    build_churn_labels,
    compute_features_for_all_customers,
)
from src.logging_utils import setup_logging

logger = setup_logging("promote_model")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

INACTIVITY_WINDOW = int(os.environ.get("INACTIVITY_WINDOW", "30"))
TOP_K_PCT = 0.20

# Primary metric used for promotion decisions — higher is always better for both.
PRIMARY_METRIC = {
    "Churn_model": "roc_auc",
    "Survival_model": "concordance_index",
}

# Optional secondary/business metric — must not regress if present for both
# candidate and prod, on top of the primary-metric check.
SECONDARY_METRIC = {
    "Churn_model": "capture_at_20pct",
}


def _metric_for_alias(
    client: MlflowClient, model_name: str, alias: str, metric_name: str
) -> float | None:
    try:
        mv = client.get_model_version_by_alias(model_name, alias)
    except Exception:
        return None
    run = client.get_run(mv.run_id)
    return run.data.metrics.get(metric_name)


# ---------------------------------------------------------------------------
# Apples-to-apples fresh-holdout comparison (Churn_model only)
# ---------------------------------------------------------------------------

def _build_fresh_holdout(transactions_file: str) -> pd.DataFrame:
    """One identical evaluation set, built now, from current data — the same
    point-in-time-safe feature path every other consumer uses. Reserves the
    trailing INACTIVITY_WINDOW days so the label has a real forward window
    to observe (see the identical fix in evaluate_model.py)."""
    transactions = pd.read_csv(transactions_file)
    transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])
    snapshot_date = transactions["transaction_date"].max() - pd.Timedelta(days=INACTIVITY_WINDOW)
    features = compute_features_for_all_customers(transactions, snapshot_date)
    labels = build_churn_labels(transactions, snapshot_date, INACTIVITY_WINDOW)
    return features.merge(labels, on="customer_id")


def _score_on_holdout(model_uri: str, holdout: pd.DataFrame) -> dict:
    model = mlflow.sklearn.load_model(model_uri)
    X = holdout[FEATURE_COLUMNS]
    y = holdout["churn"].to_numpy()
    proba = model.predict_proba(X)[:, 1]

    df = pd.DataFrame({"y": y, "score": proba}).sort_values("score", ascending=False)
    k = max(1, int(len(df) * TOP_K_PCT))
    return {
        "roc_auc": roc_auc_score(y, proba),
        "capture_at_20pct": float(df.head(k)["y"].sum() / df["y"].sum()),
    }


def _promote_churn_fresh(client: MlflowClient, model_name: str, transactions_file: str) -> bool:
    try:
        candidate_mv = client.get_model_version_by_alias(model_name, "candidate")
    except Exception as exc:
        raise RuntimeError(
            f"No 'candidate' version found for '{model_name}'. Run training first."
        ) from exc

    holdout = _build_fresh_holdout(transactions_file)
    candidate_scores = _score_on_holdout(f"models:/{model_name}@candidate", holdout)

    try:
        prod_mv = client.get_model_version_by_alias(model_name, "prod")
    except Exception:
        prod_mv = None

    if prod_mv is None:
        logger.info(
            "No existing 'prod' for '%s' — promoting candidate v%s unconditionally "
            "(fresh holdout, n=%d: roc_auc=%.4f, capture_at_20pct=%.4f).",
            model_name, candidate_mv.version, len(holdout),
            candidate_scores["roc_auc"], candidate_scores["capture_at_20pct"],
        )
        promoted = True
    else:
        prod_scores = _score_on_holdout(f"models:/{model_name}@prod", holdout)
        logger.info(
            "Apples-to-apples on ONE fresh holdout (n=%d): candidate v%s "
            "roc_auc=%.4f capture_at_20pct=%.4f | prod v%s roc_auc=%.4f capture_at_20pct=%.4f",
            len(holdout), candidate_mv.version, candidate_scores["roc_auc"], candidate_scores["capture_at_20pct"],
            prod_mv.version, prod_scores["roc_auc"], prod_scores["capture_at_20pct"],
        )
        if candidate_scores["roc_auc"] > prod_scores["roc_auc"]:
            promoted = True
        else:
            logger.info(
                "Candidate v%s does not beat prod v%s on fresh roc_auc (%.4f <= %.4f) — keeping current prod.",
                candidate_mv.version, prod_mv.version, candidate_scores["roc_auc"], prod_scores["roc_auc"],
            )
            promoted = False

        if promoted and candidate_scores["capture_at_20pct"] < prod_scores["capture_at_20pct"]:
            logger.info(
                "Candidate v%s beats prod v%s on roc_auc but regresses on capture_at_20pct "
                "(%.4f < %.4f), on the SAME fresh holdout — withholding promotion.",
                candidate_mv.version, prod_mv.version,
                candidate_scores["capture_at_20pct"], prod_scores["capture_at_20pct"],
            )
            promoted = False

    if promoted:
        client.set_registered_model_alias(model_name, "prod", candidate_mv.version)

    return promoted


# ---------------------------------------------------------------------------
# Fallback: compare each alias's own historically logged metric
# ---------------------------------------------------------------------------

def _promote_logged_metric(client: MlflowClient, model_name: str) -> bool:
    metric_name = PRIMARY_METRIC[model_name]

    candidate_metric = _metric_for_alias(client, model_name, "candidate", metric_name)
    if candidate_metric is None:
        raise RuntimeError(
            f"No 'candidate' version found for '{model_name}'. Run training first."
        )

    prod_metric = _metric_for_alias(client, model_name, "prod", metric_name)

    if prod_metric is None:
        logger.info(
            "No existing 'prod' version for '%s' — promoting candidate unconditionally (%s=%.4f).",
            model_name, metric_name, candidate_metric,
        )
        promoted = True
    elif candidate_metric > prod_metric:
        logger.info(
            "Candidate beats prod for '%s' (%s: %.4f > %.4f) — promoting.",
            model_name, metric_name, candidate_metric, prod_metric,
        )
        promoted = True
    else:
        logger.info(
            "Candidate does not beat prod for '%s' (%s: %.4f <= %.4f) — keeping current prod.",
            model_name, metric_name, candidate_metric, prod_metric,
        )
        promoted = False

    # Business-metric gate: only relevant when the primary metric already
    # says "promote" and there's an existing prod version to regress against.
    secondary_name = SECONDARY_METRIC.get(model_name)
    if promoted and prod_metric is not None and secondary_name:
        candidate_secondary = _metric_for_alias(client, model_name, "candidate", secondary_name)
        prod_secondary = _metric_for_alias(client, model_name, "prod", secondary_name)
        if candidate_secondary is None or prod_secondary is None:
            logger.warning(
                "Business metric '%s' missing for candidate and/or prod — run "
                "scripts/evaluate_business_metrics.py first for a full gate. "
                "Skipping business-metric check, promotion decided by '%s' alone.",
                secondary_name, metric_name,
            )
        elif candidate_secondary < prod_secondary:
            logger.info(
                "Candidate beats prod on '%s' but regresses on business metric "
                "'%s' (%.4f < %.4f) — withholding promotion.",
                metric_name, secondary_name, candidate_secondary, prod_secondary,
            )
            promoted = False
        else:
            logger.info(
                "Candidate also holds/improves business metric '%s' (%.4f >= %.4f).",
                secondary_name, candidate_secondary, prod_secondary,
            )

    if promoted:
        candidate_version = client.get_model_version_by_alias(model_name, "candidate").version
        client.set_registered_model_alias(model_name, "prod", candidate_version)

    return promoted


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def promote(model_name: str, transactions_file: str | None = None) -> bool:
    if model_name not in PRIMARY_METRIC:
        raise ValueError(f"Unknown model_name '{model_name}'. Expected one of {list(PRIMARY_METRIC)}.")

    client = MlflowClient()

    if model_name == "Churn_model" and transactions_file:
        return _promote_churn_fresh(client, model_name, transactions_file)

    if model_name == "Churn_model" and not transactions_file:
        logger.warning(
            "No transactions_csv given — falling back to comparing each alias's own logged "
            "metric (NOT apples-to-apples; this is exactly what let a leaky model get stuck "
            "as prod once). Pass a transactions CSV for the fresh-holdout comparison."
        )

    return _promote_logged_metric(client, model_name)


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        logger.error("Usage: python promote_model.py <Churn_model|Survival_model> [transactions_csv]")
        sys.exit(1)

    txn_file = sys.argv[2] if len(sys.argv) == 3 else None

    # Promoted or not is a valid outcome either way — exit 0 in both cases.
    # Only a missing candidate / bad model name is a real failure.
    promote(sys.argv[1], txn_file)
    sys.exit(0)
