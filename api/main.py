"""FastAPI service backed by the same single-record predictor as Streamlit."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from src.inference import ECGPredictor, load_predictor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs/checkpoints/baseline_best.pt"


class PredictionRequest(BaseModel):
    """One ten-second, twelve-lead physical ECG in samples-by-leads order."""

    model_config = ConfigDict(extra="forbid")

    signal: list[list[float]] = Field(min_length=1000, max_length=5000)
    sampling_rate_hz: int = Field(gt=0)
    lead_names: list[str] = Field(min_length=12, max_length=12)
    units: list[str] = Field(min_length=12, max_length=12)


class PredictionResponse(BaseModel):
    probabilities: dict[str, float]
    thresholds: dict[str, float]
    positive_labels: list[str]


def create_app(checkpoint_path: str | Path = DEFAULT_CHECKPOINT) -> FastAPI:
    """Build an app that loads the model once and reports missing models clearly."""
    checkpoint_path = Path(checkpoint_path)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.predictor = None
        application.state.model_error = None
        try:
            application.state.predictor = load_predictor(checkpoint_path)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            application.state.model_error = str(exc)
        yield
        application.state.predictor = None

    service = FastAPI(
        title="PTB-XL ECG Classification API",
        description="Educational five-label ECG classification; not for clinical diagnosis.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @service.get("/health")
    def health(request: Request) -> dict:
        predictor: ECGPredictor | None = request.app.state.predictor
        return {
            "status": "ok" if predictor is not None else "model_unavailable",
            "model_loaded": predictor is not None,
            "device": str(predictor.device) if predictor is not None else None,
            "model_error": request.app.state.model_error,
        }

    @service.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
        predictor: ECGPredictor | None = request.app.state.predictor
        if predictor is None:
            raise HTTPException(status_code=503, detail=request.app.state.model_error or "Model unavailable")
        try:
            result = predictor.predict(
                np.asarray(payload.signal, dtype=np.float32),
                sampling_rate_hz=payload.sampling_rate_hz,
                lead_names=payload.lead_names,
                units=payload.units,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return PredictionResponse(
            probabilities=result.probabilities,
            thresholds=result.thresholds,
            positive_labels=list(result.positive_labels),
        )

    return service


app = create_app()
