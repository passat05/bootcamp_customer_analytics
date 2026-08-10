import numpy as np
import pandas as pd

class SurvivalAnalyzer:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.model = None
        
    def initialize(self):
        """Initialize survival model"""
        # Looking for 'coxph' specifically as requested
        self.model = self.data_loader.models.get("coxph")
        if self.model is None:
            print("⚠️ Warning: 'coxph' model not found in data_loader.")
        
    def predict_survival(self, customer_id: str) -> dict:
        """Predict survival curve for a customer"""
        if self.model is None:
            raise ValueError("Model not initialized. Call .initialize() first.")
        
        # 1. Extract features for the specific ID
        features = self._extract_features(customer_id)

        # 2. Generate survival probabilities at specific checkpoints
        days = [30, 60, 90, 120, 180, 365]
        
        # Lifelines predict_survival_function returns a DF with times as index
        curve = self.model.predict_survival_function(features, times=days)

        # 3. Format into a clean list of dictionaries
        survival_curve = [
            {"day": int(d), "prob": round(float(p), 4)} 
            for d, p in curve.iloc[:, 0].items()
        ]
        
        return {
            "customer_id": customer_id,
            "survival_curve": survival_curve
        }
    
    def _extract_features(self, customer_id: str) -> pd.DataFrame:
        """
        Retrieves raw data from data_loader and transforms it into 
        the feature format expected by CoxPH.
        """
        # Fetch data for the specific customer
        customer_data = self.data_loader.get_customer_data(customer_id)
        txns = customer_data.get("transactions", pd.DataFrame())
        
        if txns.empty:
            return pd.DataFrame({
                "frequency": [0],
                "avg_amount": [0.0]
            })
        
        # Calculate features used during model training
        frequency = len(txns)
        avg_amount = txns["amount"].mean()
        
        return pd.DataFrame({
            "frequency": [frequency],
            "avg_amount": [avg_amount]
        })