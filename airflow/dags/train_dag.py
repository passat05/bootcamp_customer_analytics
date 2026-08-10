"""
train_dag.py – Retrains both models against the latest processed data, then
promotes each to the "prod" MLflow alias only if it beats the currently
served prod model (see scripts/promote_model.py).

Deliberately time-based (weekly) rather than dataset-triggered: ingestion
(simulate_data) and transform (feature_transform) run daily since new data
arrives daily, but retraining on every single day's incremental update
would be wasteful (a burst backfill of N days would trigger N retraining
runs) and isn't how retraining cadence should track ingestion cadence in
practice. feature_transform's PROCESSED_DATA output is still there whenever
this DAG runs — it just doesn't need to fire in lockstep with it.
"""

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="train_and_promote",
    description="Train churn + survival models on the latest processed data and promote winners to prod",
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["training", "mlops"],
) as dag:
    train_churn = BashOperator(
        task_id="train_churn_model",
        bash_command=(
            "python /opt/airflow/scripts/train_model.py "
            "--dataset /opt/airflow/data/processed/churn_dataset.csv"
        ),
    )

    train_survival = BashOperator(
        task_id="train_survival_model",
        bash_command=(
            "python /opt/airflow/scripts/train_survival_model.py "
            "--dataset /opt/airflow/data/processed/survival_dataset.csv"
        ),
    )

    promote_churn = BashOperator(
        task_id="promote_churn_model",
        bash_command=(
            "python /opt/airflow/scripts/promote_model.py Churn_model "
            "/opt/airflow/data/transactions.csv"
        ),
    )

    promote_survival = BashOperator(
        task_id="promote_survival_model",
        bash_command="python /opt/airflow/scripts/promote_model.py Survival_model",
    )

    train_churn >> promote_churn
    train_survival >> promote_survival
