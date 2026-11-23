"""Request / response models for the API."""
import math
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from src.config import FEATURE_NAMES


class Patient(BaseModel):
    """The 30 measurements of one tumour, keyed by the Kaggle column names (e.g. radius_mean)."""
    features: Dict[str, float] = Field(..., description="All 30 features, e.g. {'radius_mean': 17.99, ...}")

    @field_validator("features")
    @classmethod
    def all_features_present(cls, v):
        missing = [f for f in FEATURE_NAMES if f not in v]
        if missing:
            raise ValueError("Missing features: " + ", ".join(missing))
        bad = [f for f in FEATURE_NAMES if not math.isfinite(v[f])]
        if bad:
            raise ValueError("Non-finite values in: " + ", ".join(bad))
        return v


class Prediction(BaseModel):
    prediction: str
    p_malignant: float = Field(..., ge=0, le=1)
    p_benign: float = Field(..., ge=0, le=1)
    threshold: float


class BatchRequest(BaseModel):
    patients: List[Patient] = Field(..., min_length=1, max_length=5000)


class BatchItem(BaseModel):
    id: int
    prediction: str
    p_malignant: float
    p_benign: float


class BatchResponse(BaseModel):
    n_patients: int
    n_malignant: int
    threshold: float
    results: List[BatchItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    detail: Optional[str] = None
