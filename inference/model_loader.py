import os
from functools import lru_cache
from pathlib import Path

import joblib

MODEL_PATH = os.getenv("MODEL_PATH", "/app/model/model.pkl")


class ModelLoadError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_model(model_path: str = MODEL_PATH):
    path = Path(model_path)
    if not path.exists():
        raise ModelLoadError(f"Model file not found at {path}")

    try:
        model = joblib.load(path)
    except Exception as exc:
        raise ModelLoadError(f"Failed to load model from {path}: {exc}") from exc

    if not hasattr(model, "predict"):
        raise ModelLoadError("Loaded object is not a valid sklearn model (missing predict method)")

    return model
