"""
compare_boosting_models.py – Controlled LightGBM-vs-XGBoost comparison.

This project chose LightGBM at setup time without ever benchmarking it
against XGBoost. This script answers the question directly: train both on
the IDENTICAL stacked-cohort dataset, the IDENTICAL train/in-time/OOT split
(same GroupShuffleSplit seed), and comparable hyperparameter budgets (same
n_estimators/learning_rate/max_depth/scale_pos_weight — this is a same-
budget comparison, not an exhaustive tuning bake-off for either library),
then report OOT ROC-AUC, decile-1 lift, and capture@20% for both.

Diagnostic only — does not touch the MLflow model registry or the
"candidate"/"prod" aliases. Churn_model in production stays LightGBM
regardless of the result; this is here to make that choice an informed one.
"""

import os
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.feature_engineering import FEATURE_COLUMNS, build_churn_training_dataset_stacked
from src.logging_utils import setup_logging

logger = setup_logging("compare_boosting_models")

INACTIVITY_WINDOW = int(os.environ.get("INACTIVITY_WINDOW", "30"))
TOP_K_PCT = 0.20


def decile1_lift(y_true: np.ndarray, y_score: np.ndarray) -> float:
    df = pd.DataFrame({"y": y_true, "score": y_score})
    df["decile"] = 10 - pd.qcut(df["score"], 10, labels=False, duplicates="drop")
    overall_rate = df["y"].mean()
    decile1_rate = df.loc[df["decile"] == 1, "y"].mean()
    return decile1_rate / overall_rate


def capture_at_k(y_true: np.ndarray, y_score: np.ndarray, k_pct: float) -> float:
    df = pd.DataFrame({"y": y_true, "score": y_score}).sort_values("score", ascending=False)
    k = max(1, int(len(df) * k_pct))
    return float(df.head(k)["y"].sum() / df["y"].sum())


def run(transactions_file: str, snapshot_freq_days: int = 30, min_history_days: int = 90) -> pd.DataFrame:
    transactions = pd.read_csv(transactions_file)
    transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])

    stacked = build_churn_training_dataset_stacked(
        transactions, INACTIVITY_WINDOW,
        snapshot_freq_days=snapshot_freq_days, min_history_days=min_history_days,
    )
    oot_snapshot = stacked["snapshot_date"].max()
    oot = stacked[stacked["snapshot_date"] == oot_snapshot]
    pool = stacked[stacked["snapshot_date"] != oot_snapshot]

    # Identical split for both models — same customers in train/in-time for
    # both, so any AUC difference is attributable to the algorithm alone.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, intime_idx = next(gss.split(pool, groups=pool["customer_id"]))
    train_df, intime_df = pool.iloc[train_idx], pool.iloc[intime_idx]

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["churn"]
    X_intime, y_intime = intime_df[FEATURE_COLUMNS], intime_df["churn"]
    X_oot, y_oot = oot[FEATURE_COLUMNS], oot["churn"]

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos
    logger.info("Snapshot OOT: %s | train=%d, in-time=%d, OOT=%d | scale_pos_weight=%.3f",
                oot_snapshot.date(), len(train_df), len(intime_df), len(oot), scale_pos_weight)

    # Same-budget hyperparameters — comparable capacity, not tuned per library.
    models = {
        "LightGBM": LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42,
            n_jobs=-1, scale_pos_weight=scale_pos_weight, verbosity=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42,
            n_jobs=-1, scale_pos_weight=scale_pos_weight,
            eval_metric="auc", use_label_encoder=False, verbosity=0,
        ),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)

        proba_intime = model.predict_proba(X_intime)[:, 1]
        proba_oot = model.predict_proba(X_oot)[:, 1]

        rows.append({
            "model": name,
            "intime_auc": roc_auc_score(y_intime, proba_intime),
            "oot_auc": roc_auc_score(y_oot, proba_oot),
            "oot_decile1_lift": decile1_lift(y_oot.to_numpy(), proba_oot),
            "oot_capture_at_20pct": capture_at_k(y_oot.to_numpy(), proba_oot, TOP_K_PCT),
        })
        logger.info("%s — in-time AUC=%.4f, OOT AUC=%.4f", name,
                    rows[-1]["intime_auc"], rows[-1]["oot_auc"])

    results = pd.DataFrame(rows).set_index("model")
    results["intime_vs_oot_gap"] = results["intime_auc"] - results["oot_auc"]
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Usage: python compare_boosting_models.py <transactions_csv>")
        sys.exit(1)

    results = run(sys.argv[1])
    logger.info("Comparison results:\n%s", results.to_string())

    os.makedirs("reports", exist_ok=True)
    out_path = "reports/lightgbm_vs_xgboost.csv"
    results.to_csv(out_path)
    logger.info("Written to %s", out_path)
    sys.exit(0)
