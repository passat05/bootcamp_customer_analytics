"""
data_loader.py – Production-ready data and model loader.

Models are loaded directly from the MLflow Model Registry by alias
(e.g. "Churn_model@prod"), not from static .pkl files. This guarantees
the API always serves whatever model was most recently promoted to
"prod" via train_model.py — no manual file copying required.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import mlflow

from src.schemas import CustomerRecord, TransactionRecord, validate_dataframe

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Map internal model name → (registered_model_name, alias)
MODEL_REGISTRY_MAP = {
    "churn_classification": ("Churn_model", "prod"),
    "coxph": ("Survival_model", "prod"),
}


class DataLoader:
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.customers: pd.DataFrame | None = None
        self.transactions: pd.DataFrame | None = None
        self.models: Dict[str, Any] = {}

        self._customer_index: Dict[str, dict] = {}
        self._txn_index: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_data(self) -> None:
        """Load customer and transaction CSVs and build look-up indexes."""
        customers_path = self.base_path / "data" / "customers.csv"
        transactions_path = self.base_path / "data" / "transactions.csv"

        for p in (customers_path, transactions_path):
            if not p.exists():
                raise FileNotFoundError(f"Required data file not found: {p}")

        self.customers = pd.read_csv(customers_path)
        self.transactions = pd.read_csv(transactions_path)

        validate_dataframe(self.customers, CustomerRecord, str(customers_path))
        validate_dataframe(self.transactions, TransactionRecord, str(transactions_path))

        dup_ids = self.customers["customer_id"][self.customers["customer_id"].duplicated()]
        if not dup_ids.empty:
            raise ValueError(
                f"customers.csv contains duplicate customer_id values: "
                f"{sorted(set(dup_ids))[:10]}"
            )

        self.transactions["transaction_date"] = pd.to_datetime(
            self.transactions["transaction_date"], utc=False
        )
        self.customers["signup_date"] = pd.to_datetime(
            self.customers["signup_date"], utc=False
        )

        self._customer_index = self.customers.set_index("customer_id").to_dict("index")
        self._txn_index = {
            cid: grp.reset_index(drop=True)
            for cid, grp in self.transactions.groupby("customer_id")
        }

        logger.info(
            "Data loaded: %d customers, %d transactions.",
            len(self.customers),
            len(self.transactions),
        )

    def load_models(self) -> None:
        """Load all models from the MLflow Model Registry by alias."""
        for internal_name, (registered_name, alias) in MODEL_REGISTRY_MAP.items():
            model_uri = f"models:/{registered_name}@{alias}"
            try:
                self.models[internal_name] = mlflow.sklearn.load_model(model_uri)
                logger.info("Loaded model '%s' from %s", internal_name, model_uri)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not load model '%s' from %s: %s. "
                    "Make sure train_model.py has been run and registered this alias.",
                    internal_name, model_uri, exc,
                )

    def get_customer_data(self, customer_id: str) -> Dict[str, Any]:
        if self.customers is None:
            raise RuntimeError("Call load_data() before get_customer_data().")

        if customer_id not in self._customer_index:
            raise KeyError(f"Customer '{customer_id}' not found.")

        return {
            "customer": self._customer_index[customer_id],
            "transactions": self._txn_index.get(customer_id, pd.DataFrame()),
        }