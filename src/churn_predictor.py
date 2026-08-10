"""
churn_predictor.py – Production-ready churn prediction class.

Feature computation is delegated entirely to feature_engineering.py,
which is the single source of truth shared with train_model.py.
This guarantees training-serving consistency.
"""

import logging
from datetime import datetime
from typing import Dict, Any

import pandas as pd

from src.feature_engineering import (
    FEATURE_COLUMNS,
    compute_features_for_customer,
)

logger = logging.getLogger(__name__)

HIGH_RISK_THRESHOLD   = 0.70
MEDIUM_RISK_THRESHOLD = 0.40


class ChurnPredictor:
    def __init__(self, data_loader, snapshot_date: datetime | None = None):
        """
        Parameters
        ----------
        data_loader : DataLoader
        snapshot_date : datetime, optional
            Reference date for RFM calculation. Defaults to utcnow().
            Pass an explicit date during backtesting or unit tests.
        """
        self.data_loader = data_loader
        self.snapshot_date = pd.Timestamp(snapshot_date or datetime.utcnow())
        self.model = None

    def initialize(self) -> None:
        self.model = self.data_loader.models.get("churn_classification")
        if self.model is None:
            logger.warning(
                "Model 'churn_classification' not found. "
                "Calls to predict_churn() will raise until a model is loaded."
            )

    def predict_churn(self, customer_id: str) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError(
                "ChurnPredictor has no model. Call initialize() after loading models."
            )

        customer_data = self.data_loader.get_customer_data(customer_id)
        txns: pd.DataFrame = customer_data.get("transactions", pd.DataFrame())

        # ✅ Shared feature computation — same as training
        features = compute_features_for_customer(txns, self.snapshot_date)

        churn_prob: float = float(self.model.predict_proba(features)[0, 1])
        label = self._classify(churn_prob)

        logger.debug("customer=%s  prob=%.4f  label=%s", customer_id, churn_prob, label)

        return {
            "customer_id":       customer_id,
            "churn_probability": round(churn_prob, 4),
            "churn_label":       label,
        }

    def _classify(self, prob: float) -> str:
        if prob >= HIGH_RISK_THRESHOLD:
            return "high_risk"
        if prob >= MEDIUM_RISK_THRESHOLD:
            return "medium_risk"
        return "low_risk"