from typing import Annotated, List, Literal

from pydantic import BaseModel, ConfigDict, Field

# Shared customer_id constraint: non-empty after trimming, bounded length.
CustomerId = Annotated[str, Field(min_length=1, max_length=64)]


class _StrictModel(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)


class CustomerScoreRequest(_StrictModel):
    customer_id: CustomerId

class ChurnPredictionRequest(_StrictModel):
    customer_id: CustomerId
    horizon_days: int = Field(default=30, gt=0, le=365)

class ChurnPredictionResponse(_StrictModel):
    customer_id: CustomerId
    churn_probability: float = Field(ge=0.0, le=1.0)
    churn_label: Literal["low_risk", "medium_risk", "high_risk"]

class SurvivalPoint(_StrictModel):
    day: int = Field(gt=0)
    prob: float = Field(ge=0.0, le=1.0)

class SurvivalPredictionRequest(_StrictModel):
    customer_id: CustomerId

class SurvivalPredictionResponse(_StrictModel):
    customer_id: CustomerId
    survival_curve: List[SurvivalPoint]