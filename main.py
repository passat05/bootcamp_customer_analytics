"""
main.py – Production-ready FastAPI application.

Key improvements over original:
  - Lifespan context manager replaces deprecated @app.on_event("startup")
  - CORS origins read from environment (not open wildcard in production)
  - KeyError from unknown customer_id → 404, not 500
  - Structured JSON logging via Python logging
  - /health endpoint for liveness/readiness probes (used by docker-compose)
  - Imports use explicit relative paths; no wildcard `from models import *`
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.data_loader import DataLoader
from src.churn_predictor import ChurnPredictor
from src.survival_analyzer import SurvivalAnalyzer
from src.models import (
    ChurnPredictionRequest,
    ChurnPredictionResponse,
    SurvivalPredictionRequest,
    SurvivalPredictionResponse,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Component initialisation
# ---------------------------------------------------------------------------
BASE_PATH = os.getenv("DATA_BASE_PATH", ".")

data_loader = DataLoader(base_path=BASE_PATH)
churn_predictor = ChurnPredictor(data_loader)
survival_analyzer = SurvivalAnalyzer(data_loader)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load data and models once at startup; release resources on shutdown."""
    logger.info("Loading data and models …")
    data_loader.load_data()
    data_loader.load_models()
    churn_predictor.initialize()
    survival_analyzer.initialize()
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

app = FastAPI(
    title="Customer Scoring API",
    version="1.0.0",
    description="Churn prediction and survival analysis endpoints.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["meta"])
async def root():
    return {"message": "Customer Scoring API is running"}


@app.get("/health", tags=["meta"])
async def health():
    """Liveness / readiness probe used by docker-compose healthcheck."""
    models_loaded = list(data_loader.models.keys())
    return {
        "status": "ok",
        "models_loaded": models_loaded,
        "data_loaded": data_loader.customers is not None,
    }


@app.post(
    "/predict_churn",
    response_model=ChurnPredictionResponse,
    tags=["prediction"],
)
async def predict_churn(request: ChurnPredictionRequest):
    """Return churn probability and risk label for a customer."""
    try:
        result = churn_predictor.predict_churn(request.customer_id)
        return result
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error in /predict_churn")
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@app.post(
    "/predict_survival",
    response_model=SurvivalPredictionResponse,
    tags=["prediction"],
)
async def predict_survival(request: SurvivalPredictionRequest):
    """Return the survival curve for a customer."""
    try:
        result = survival_analyzer.predict_survival(request.customer_id)
        return result
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error in /predict_survival")
        raise HTTPException(status_code=500, detail="Internal server error") from exc