"""
feature_engineering.py – Single source of truth for all feature computation.

Both train_model.py (training) and ChurnPredictor (serving) import from here.
This eliminates training-serving skew: the exact same feature logic is used
at train time and at inference time.

Feature set: RFM_v1
  recency_days  – days since last transaction
  freq_30d      – number of transactions in last 30 days
  freq_90d      – number of transactions in last 90 days
  monetary_90d  – total spend in last 90 days
  freq_ratio    – freq_30d / freq_90d  (captures fading behaviour)
"""

import logging
from datetime import datetime, timedelta
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Single definition of feature columns — imported everywhere
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: List[str] = [
    "recency_days",
    "freq_30d",
    "freq_90d",
    "monetary_90d",
    "freq_ratio",
]

# Sentinel row used when a customer has no transactions at all
EMPTY_FEATURE_ROW = [999, 0, 0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Core computation (pure functions — easy to unit-test)
# ---------------------------------------------------------------------------

def compute_features_for_customer(
    txns: pd.DataFrame,
    snapshot_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Compute RFM_v1 features for a single customer's transaction history.

    Parameters
    ----------
    txns : pd.DataFrame
        Must contain columns: transaction_date (datetime64), amount (float).
        May be empty — returns a zero-filled sentinel row in that case.
    snapshot_date : pd.Timestamp
        Reference date for all time-window calculations.

    Returns
    -------
    pd.DataFrame  shape (1, 5)  with columns = FEATURE_COLUMNS
    """
    # Drop any transaction after the snapshot — at live-serving time this is a
    # no-op (there's no such thing as a future transaction), but it keeps this
    # function safe to call during backtesting with an explicit historical
    # snapshot_date, per the docstring above.
    txns = txns[txns["transaction_date"] <= snapshot_date]

    if txns.empty:
        logger.warning("Empty transaction history — using zero-fill features.")
        return pd.DataFrame([EMPTY_FEATURE_ROW], columns=FEATURE_COLUMNS)

    d30_cutoff = snapshot_date - timedelta(days=30)
    d90_cutoff = snapshot_date - timedelta(days=90)

    last_txn = txns["transaction_date"].max()
    recency_days = int((snapshot_date - last_txn).days)

    freq_30d = int((txns["transaction_date"] >= d30_cutoff).sum())
    mask_90 = txns["transaction_date"] >= d90_cutoff
    freq_90d = int(mask_90.sum())
    monetary_90d = float(txns.loc[mask_90, "amount"].sum())
    freq_ratio = freq_30d / (freq_90d + 1e-6)

    return pd.DataFrame(
        [[recency_days, freq_30d, freq_90d, monetary_90d, freq_ratio]],
        columns=FEATURE_COLUMNS,
    )


def compute_features_for_all_customers(
    transactions: pd.DataFrame,
    snapshot_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Compute RFM_v1 features for every customer in bulk (used during training).

    Parameters
    ----------
    transactions : pd.DataFrame
        Full transaction table with columns:
        customer_id, transaction_date (datetime64), amount (float).
    snapshot_date : pd.Timestamp
        Reference date for all time-window calculations.

    Returns
    -------
    pd.DataFrame  with columns: customer_id + FEATURE_COLUMNS
    """
    # Drop any transaction after the snapshot — without this, a customer whose
    # last purchase falls in the trailing (label) window leaks future
    # information into recency/frequency/monetary features.
    transactions = transactions[transactions["transaction_date"] <= snapshot_date]

    d30_cutoff = snapshot_date - timedelta(days=30)
    d90_cutoff = snapshot_date - timedelta(days=90)

    def _per_customer(g: pd.DataFrame) -> pd.Series:
        last_txn = g["transaction_date"].max()
        freq_30d = int((g["transaction_date"] >= d30_cutoff).sum())
        mask_90 = g["transaction_date"] >= d90_cutoff
        freq_90d = int(mask_90.sum())
        monetary_90d = float(g.loc[mask_90, "amount"].sum())
        return pd.Series({
            "recency_days": int((snapshot_date - last_txn).days),
            "freq_30d":     freq_30d,
            "freq_90d":     freq_90d,
            "monetary_90d": monetary_90d,
            "freq_ratio":   freq_30d / (freq_90d + 1e-6),
        })

    features = (
        transactions.groupby("customer_id")
        .apply(_per_customer)
        .reset_index()
    )

    logger.info("Computed features for %d customers.", len(features))
    return features


def compute_snapshot_date(transactions: pd.DataFrame) -> pd.Timestamp:
    """
    80%-of-timeline snapshot date shared by every training/transform entry
    point, so a label is never computed with information from beyond the
    snapshot (the trailing 20% of the timeline is reserved to observe
    whether a customer churns after the snapshot).
    """
    min_date = transactions["transaction_date"].min()
    max_date = transactions["transaction_date"].max()
    return min_date + (max_date - min_date) * 0.8


def build_churn_training_dataset(
    transactions: pd.DataFrame,
    inactivity_window: int,
) -> pd.DataFrame:
    """Build the (customer_id + FEATURE_COLUMNS + churn) table used to train/retrain the classifier."""
    snapshot_date = compute_snapshot_date(transactions)
    features = compute_features_for_all_customers(transactions, snapshot_date)
    labels = build_churn_labels(transactions, snapshot_date, inactivity_window)
    return features.merge(labels, on="customer_id")


def build_survival_training_dataset(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    inactivity_window: int,
    snapshot_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Build the (customer_id, frequency, avg_amount, duration, event) table
    used to fit CoxPH. Mirrors build_churn_training_dataset's snapshot logic
    so "event=1" here means the same thing as "churn=1" there.

    snapshot_date defaults to compute_snapshot_date(transactions) (the
    existing 80%-of-timeline behavior, unchanged for train_survival_model.py)
    but can be overridden so a caller can build this dataset against the
    same snapshot used elsewhere (e.g. to compare against a churn model's
    OOT cohort).
    """
    if snapshot_date is None:
        snapshot_date = compute_snapshot_date(transactions)
    labels = build_churn_labels(transactions, snapshot_date, inactivity_window)
    labels = labels.rename(columns={"churn": "event"})

    txns_observed = transactions[transactions["transaction_date"] <= snapshot_date]

    # Point-in-time discipline: frequency/avg_amount must only reflect what
    # happened on/before snapshot_date — otherwise a customer's post-snapshot
    # activity leaks into the covariates used to explain their own event.
    covariates = txns_observed.groupby("customer_id").agg(
        frequency=("amount", "count"),
        avg_amount=("amount", "mean"),
    ).reset_index()

    last_txn_observed = (
        txns_observed.groupby("customer_id")["transaction_date"].max()
        .rename("last_txn_observed")
    )

    data = (
        covariates
        .merge(labels, on="customer_id")
        .merge(customers[["customer_id", "signup_date"]], on="customer_id")
        .merge(last_txn_observed, on="customer_id", how="left")
    )

    # Customers with no transactions on/before the snapshot: treat as censored at snapshot.
    data["last_txn_observed"] = data["last_txn_observed"].fillna(snapshot_date)

    reference_date = data["last_txn_observed"].where(data["event"] == 1, snapshot_date)
    data["duration"] = (reference_date - data["signup_date"]).dt.days.clip(lower=1)

    return data[["customer_id", "frequency", "avg_amount", "duration", "event"]]


def build_churn_training_dataset_stacked(
    transactions: pd.DataFrame,
    inactivity_window: int,
    snapshot_freq_days: int = 30,
    min_history_days: int = 90,
) -> pd.DataFrame:
    """
    Stacked-cohorts variant of build_churn_training_dataset (see the
    Propensity Model methodology, "kỹ thuật stacked cohorts"): instead of a
    single 80%-of-timeline snapshot, generate one cohort every
    `snapshot_freq_days` across the observable window. Each cohort
    contributes its own (customer_id, snapshot_date, features..., churn)
    rows, all computed with the same point-in-time discipline as
    compute_features_for_all_customers / build_churn_labels.

    min_history_days keeps the earliest snapshot far enough past the start
    of the timeline that recency/frequency features aren't meaningless: the
    latest snapshot still reserves inactivity_window days at the end of the
    timeline to observe the label, exactly as compute_snapshot_date does for
    the single-snapshot path.

    Rows from the same customer across different snapshots are correlated —
    callers MUST split by customer_id (group split), never by row, when
    building train/validation/test sets from this output.
    """
    min_date = transactions["transaction_date"].min()
    max_date = transactions["transaction_date"].max()

    earliest_snapshot = min_date + timedelta(days=min_history_days)
    latest_snapshot = max_date - timedelta(days=inactivity_window)

    if earliest_snapshot >= latest_snapshot:
        raise ValueError(
            f"Timeline too short for stacked cohorts: earliest_snapshot="
            f"{earliest_snapshot.date()} >= latest_snapshot={latest_snapshot.date()}. "
            "Reduce min_history_days/inactivity_window or provide more data."
        )

    snapshot_dates = pd.date_range(earliest_snapshot, latest_snapshot, freq=f"{snapshot_freq_days}D")

    cohorts = []
    for snapshot_date in snapshot_dates:
        features = compute_features_for_all_customers(transactions, snapshot_date)
        labels = build_churn_labels(transactions, snapshot_date, inactivity_window)
        cohort = features.merge(labels, on="customer_id")
        cohort.insert(1, "snapshot_date", snapshot_date)
        cohorts.append(cohort)

    stacked = pd.concat(cohorts, ignore_index=True)
    logger.info(
        "Stacked cohorts: %d snapshots (%s -> %s), %d total rows, churn rate %.1f%%",
        len(snapshot_dates), snapshot_dates[0].date(), snapshot_dates[-1].date(),
        len(stacked), stacked["churn"].mean() * 100,
    )
    return stacked


def compute_customer_value_proxy(
    transactions: pd.DataFrame,
    snapshot_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Lightweight customer-value proxy for priority scoring (propensity x
    value), NOT a trained CLV model: annualized average monthly spend over
    the customer's full observed history up to snapshot_date (same
    point-in-time discipline as every other feature in this module).

    Deliberately a lifetime average, not a short trailing window (e.g. last
    90 days): a customer who is starting to churn also stops buying
    recently by definition, so a recency-weighted value proxy collapses
    toward zero for exactly the customers the churn model flags as
    at-risk — cancelling out the risk signal in a priority score instead of
    combining with it. A lifetime average keeps "how valuable has this
    customer historically been" separate from "did they buy something this
    week," so a high-value customer who has recently gone quiet still shows
    up as high-value-at-risk rather than being scored away.

    Returns
    -------
    pd.DataFrame  with columns: customer_id, customer_value
    """
    txns = transactions[transactions["transaction_date"] <= snapshot_date]
    agg = txns.groupby("customer_id").agg(
        total_spend=("amount", "sum"),
        first_txn=("transaction_date", "min"),
    )
    tenure_months = ((snapshot_date - agg["first_txn"]).dt.days / 30.44).clip(lower=1)
    value = (agg["total_spend"] / tenure_months * 12).rename("customer_value")

    all_customers = transactions["customer_id"].unique()
    result = pd.DataFrame({"customer_id": all_customers}).merge(
        value.reset_index(), on="customer_id", how="left"
    )
    result["customer_value"] = result["customer_value"].fillna(0.0)
    return result


def build_churn_labels(
    transactions: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    inactivity_window: int,
) -> pd.DataFrame:
    """
    Label each customer as churned (1) or retained (0).

    A customer is churned if they made NO purchase in the
    (snapshot_date, snapshot_date + inactivity_window] window.

    Parameters
    ----------
    transactions : pd.DataFrame
        Full transaction table.
    snapshot_date : pd.Timestamp
        End of the observation period.
    inactivity_window : int
        Number of days in the prediction horizon.

    Returns
    -------
    pd.DataFrame  with columns: customer_id, churn
    """
    future_end = snapshot_date + timedelta(days=inactivity_window)

    active_in_future = transactions.loc[
        (transactions["transaction_date"] > snapshot_date)
        & (transactions["transaction_date"] <= future_end),
        "customer_id",
    ].unique()

    all_customers = transactions["customer_id"].unique()
    labels = pd.DataFrame({"customer_id": all_customers})
    labels["churn"] = (~labels["customer_id"].isin(active_in_future)).astype(int)

    logger.info(
        "Labels built: %d customers, churn rate %.1f%%",
        len(labels),
        labels["churn"].mean() * 100,
    )
    return labels