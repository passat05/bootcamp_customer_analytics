"""
simulate_data_dag.py – Simulates one day of production traffic (new
transactions, occasional new customer signups) landing in data/*.csv.

Stands in for a real ingestion pipeline (e.g. CDC from an OLTP database, or
an event stream) so the rest of the platform can be exercised against
continuously growing data instead of the static historical dump the
project shipped with.

Schedule: daily, starting the day after the last historical transaction
(2025-12-31), with catchup enabled — enabling this DAG backfills every
missing day up to "today" and then keeps the dataset growing in real time.
max_active_runs=1 because each run appends to the same CSV files; runs
must never race each other.
"""

from datetime import datetime

from airflow import DAG
from airflow.datasets import Dataset
from airflow.operators.bash import BashOperator

RAW_DATA = Dataset("file:///opt/airflow/data/transactions.csv")

with DAG(
    dag_id="simulate_data",
    description="Append one simulated day of customer/transaction activity to data/*.csv",
    start_date=datetime(2026, 1, 1),  # day after the last historical transaction
    schedule="@daily",
    catchup=True,
    max_active_runs=1,
    tags=["ingestion", "simulation"],
) as dag:
    simulate = BashOperator(
        task_id="simulate_daily_activity",
        bash_command=(
            "python /opt/airflow/scripts/simulate_data.py "
            "--customers /opt/airflow/data/customers.csv "
            "--transactions /opt/airflow/data/transactions.csv "
            "--date {{ ds }}"
        ),
        outlets=[RAW_DATA],
    )
