import os
from math import isfinite
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from prometheus_fastapi_instrumentator import Instrumentator

from model_loader import MODEL_PATH, ModelLoadError, load_model

EXPECTED_FEATURES = int(os.getenv("EXPECTED_FEATURES", "20"))

app = FastAPI(
    title="Credit Risk Prediction API",
    version="1.0.0",
    description="FastAPI service for credit risk inference",
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


class PredictRequest(BaseModel):
    features: List[float] = Field(..., description=f"Feature vector with exactly {EXPECTED_FEATURES} values")

    @field_validator("features")
    @classmethod
    def validate_features(cls, values: List[float]) -> List[float]:
        if len(values) != EXPECTED_FEATURES:
            raise ValueError(f"Expected {EXPECTED_FEATURES} features, got {len(values)}")
        if not all(isfinite(v) for v in values):
            raise ValueError("All feature values must be finite numbers")
        return values


class PredictResponse(BaseModel):
    prediction: int = Field(..., ge=0, le=1)
    probability: float = Field(..., ge=0.0, le=1.0)


@app.on_event("startup")
def startup_model_check() -> None:
    load_model()


@app.get("/health")
def health() -> dict:
    try:
        load_model()
        return {"status": "ok", "model_path": MODEL_PATH}
    except ModelLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        model = load_model()
        features = np.array(payload.features, dtype=np.float64).reshape(1, -1)

        prediction = int(model.predict(features)[0])

        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(features)[0][1])
        else:
            probability = float(prediction)

        return PredictResponse(prediction=prediction, probability=probability)

    except ModelLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid input: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
