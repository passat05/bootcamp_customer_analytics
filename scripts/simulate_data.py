"""
simulate_data.py – Appends one simulated day of production activity (new
customer signups + transactions) to data/customers.csv and
data/transactions.csv.

The original CSVs are a one-time historical dump ending 2025-12-31. Run once
per simulated day (see airflow/dags/simulate_data_dag.py) to keep the
dataset growing instead of staying frozen at that cutoff.

Each customer's propensity to transact on a given day is derived from their
own historical transaction frequency, so naturally active customers keep
transacting and naturally inactive customers stay quiet — simulated days
have the same behavioral shape as the original data rather than being
uniformly random. Runs are seeded deterministically by date, so re-running
the same --date is a no-op instead of double-appending data.
"""

import argparse
import sys
from datetime import datetime

import numpy as np
import pandas as pd


def _next_customer_ids(existing_ids: pd.Series, n: int) -> list[str]:
    if n == 0:
        return []
    max_num = max(
        (int(cid[1:]) for cid in existing_ids if cid.startswith("C") and cid[1:].isdigit()),
        default=-1,
    )
    return [f"C{max_num + 1 + i:05d}" for i in range(n)]


def _append_csv(path: str, new_rows: pd.DataFrame, index_start: int) -> None:
    """Append rows, preserving the leading unnamed pandas-index column already in these CSVs."""
    if new_rows.empty:
        return
    out = new_rows.copy()
    out.insert(0, "", range(index_start, index_start + len(out)))
    out.to_csv(path, mode="a", header=False, index=False)


def simulate_day(
    customers_path: str,
    transactions_path: str,
    sim_date: pd.Timestamp,
    rng: np.random.Generator,
) -> None:
    customers = pd.read_csv(customers_path)
    customers["signup_date"] = pd.to_datetime(customers["signup_date"])

    transactions = pd.read_csv(transactions_path)
    transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])

    already_run = (
        (transactions["transaction_date"] == sim_date).any()
        or (customers["signup_date"] == sim_date).any()
    )
    if already_run:
        print(f"[simulate_data] {sim_date.date()} already simulated — skipping (idempotent).")
        return

    # ---- 1. New customer signups --------------------------------------
    history_days = max((customers["signup_date"].max() - customers["signup_date"].min()).days, 1)
    daily_signup_rate = len(customers) / history_days
    n_new = int(rng.poisson(lam=max(daily_signup_rate, 0.1)))
    new_ids = _next_customer_ids(customers["customer_id"], n_new)

    new_customers = pd.DataFrame({
        "customer_id": new_ids,
        "signup_date": [sim_date] * n_new,
        "true_lifetime_days": rng.choice(customers["true_lifetime_days"].values, size=n_new) if n_new else [],
    })

    # ---- 2. Transactions from existing customers -----------------------
    # Rate = this customer's total historical transactions / the FULL dataset
    # time window (not the customer's own min-max span) — using each
    # customer's own span would massively over-estimate activity for anyone
    # whose few transactions happened to land close together by chance.
    hist_counts = transactions.groupby("customer_id").size()
    dataset_span_days = max(
        (transactions["transaction_date"].max() - transactions["transaction_date"].min()).days, 1
    )
    daily_rate = (hist_counts / dataset_span_days).reindex(customers["customer_id"]).fillna(0.0)

    draws = rng.random(len(daily_rate))
    transacting_today = daily_rate.index[draws < daily_rate.to_numpy()]

    amount_pool = transactions["amount"].to_numpy()
    new_transactions = _make_transactions(transacting_today, sim_date, amount_pool, rng)

    # ---- 3. A fraction of today's new signups transact the same day ----
    if n_new:
        first_day_draws = rng.random(n_new)
        first_day_buyers = new_customers.loc[first_day_draws < 0.3, "customer_id"]
        extra = _make_transactions(first_day_buyers, sim_date, amount_pool, rng)
        new_transactions = pd.concat([new_transactions, extra], ignore_index=True)

    # ---- 4. Persist ------------------------------------------------------
    _append_csv(customers_path, new_customers, index_start=len(customers))
    _append_csv(transactions_path, new_transactions, index_start=len(transactions))

    print(
        f"[simulate_data] {sim_date.date()}: +{n_new} new customers, "
        f"+{len(new_transactions)} transactions."
    )


def _make_transactions(customer_ids, sim_date, amount_pool, rng) -> pd.DataFrame:
    customer_ids = pd.Index(customer_ids)
    n = len(customer_ids)
    if n == 0:
        return pd.DataFrame(columns=["customer_id", "transaction_date", "amount"])

    amounts = rng.choice(amount_pool, size=n) * rng.uniform(0.85, 1.15, size=n)
    return pd.DataFrame({
        "customer_id": customer_ids,
        "transaction_date": [sim_date] * n,
        "amount": np.round(amounts, 2),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customers", required=True, help="Path to customers.csv")
    parser.add_argument("--transactions", required=True, help="Path to transactions.csv")
    parser.add_argument("--date", required=True, help="Simulated day, YYYY-MM-DD (e.g. Airflow's {{ ds }})")
    parser.add_argument("--seed", type=int, default=None, help="Defaults to a date-derived seed for reproducibility")
    args = parser.parse_args()

    sim_date = pd.Timestamp(datetime.strptime(args.date, "%Y-%m-%d"))
    seed = args.seed if args.seed is not None else int(sim_date.strftime("%Y%m%d"))
    rng = np.random.default_rng(seed)

    simulate_day(args.customers, args.transactions, sim_date, rng)


if __name__ == "__main__":
    main()
    sys.exit(0)
