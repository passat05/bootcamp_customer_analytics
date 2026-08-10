"""
schemas.py – Pydantic schemas for validating raw CSV data at load time.

These are row-level contracts for data/customers.csv and data/transactions.csv.
Validating here means a malformed export (bad dtypes, missing customer_id,
negative amounts, unparseable dates, duplicate ids) fails loudly at startup
instead of surfacing later as a confusing KeyError or silent NaN deep inside
feature_engineering.py.
"""

import logging
from datetime import date
from typing import List, Type, TypeVar

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class CustomerRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    customer_id: str = Field(min_length=1)
    signup_date: date
    true_lifetime_days: int = Field(ge=0)


class TransactionRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    customer_id: str = Field(min_length=1)
    transaction_date: date
    amount: float = Field(gt=0)


def validate_dataframe(df: pd.DataFrame, schema: Type[T], source: str) -> None:
    """Validate every row of `df` against `schema`.

    Raises ValueError with a readable summary (first 10 offending rows) if
    any row fails validation.
    """
    adapter = TypeAdapter(List[schema])
    try:
        adapter.validate_python(df.to_dict("records"))
    except ValidationError as exc:
        errors = exc.errors()
        preview = "\n".join(
            f"  row {e['loc'][0]}: {e['loc'][1:]} — {e['msg']}" for e in errors[:10]
        )
        raise ValueError(
            f"{source} failed validation ({len(errors)} error(s), showing up to 10):\n{preview}"
        ) from exc
    logger.info("%s: %d rows passed validation.", source, len(df))
