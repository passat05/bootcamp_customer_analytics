"""
feature_transform_dag.py – Rebuilds the churn/survival training datasets
from all data-to-date whenever new raw data lands.

Data-aware scheduling: this DAG has no cron schedule of its own — it fires
automatically whenever simulate_data's task updates the RAW_DATA dataset,
so the transform always runs against the freshest data instead of guessing
a safe delay after ingestion finishes.
"""

from datetime import datetime

from airflow import DAG
from airflow.datasets import Dataset
from airflow.operators.bash import BashOperator

RAW_DATA = Dataset("file:///opt/airflow/data/transactions.csv")
PROCESSED_DATA = Dataset("file:///opt/airflow/data/processed/churn_dataset.csv")

with DAG(
    dag_id="feature_transform",
    description="Materialize churn/survival training datasets from raw customer + transaction data",
    start_date=datetime(2026, 1, 1),
    schedule=[RAW_DATA],
    catchup=False,
    tags=["transform"],
) as dag:
    build_features = BashOperator(
        task_id="build_features",
        bash_command=(
            "python /opt/airflow/scripts/build_features.py "
            "/opt/airflow/data/customers.csv "
            "/opt/airflow/data/transactions.csv "
            "/opt/airflow/data/processed"
        ),
        outlets=[PROCESSED_DATA],
    )
