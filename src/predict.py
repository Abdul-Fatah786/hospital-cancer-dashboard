"""Score patients with the saved model. Used by the Streamlit app and the FastAPI service."""
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.config import FEATURE_NAMES, MODEL_FILE, MODELS_DIR, SCALER_FILE, TARGET
from src.model import HospitalNet

_CACHE = {}


def models_dir() -> Path:
    """Folder with the saved model. Override with the HOSPITAL_MODELS_DIR environment variable."""
    return Path(os.environ.get("HOSPITAL_MODELS_DIR", MODELS_DIR))


def load_bundle(directory=None) -> dict:
    d = Path(directory) if directory else models_dir()
    key = str(d)
    if key not in _CACHE:
        mp = d / MODEL_FILE
        if not mp.exists():
            raise FileNotFoundError(f"Model not found at {mp}. Train it first with:  python -m src.train")
        meta = torch.load(mp, weights_only=True)
        model = HospitalNet(meta["n_features"], meta["hidden"], meta["dropout"])
        model.load_state_dict(meta["state_dict"])
        model.eval()
        scaler = joblib.load(d / SCALER_FILE) if meta["scaling"] else None
        _CACHE[key] = {"model": model, "scaler": scaler, "meta": meta, "threshold": float(meta["threshold"])}
    return _CACHE[key]


def validate_features(df: pd.DataFrame) -> list:
    problems = []
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        problems.append("Missing columns: " + ", ".join(missing))
    elif df[FEATURE_NAMES].isnull().any().any():
        problems.append("The feature columns contain missing values")
    return problems


def score_dataframe(df: pd.DataFrame, threshold=None, bundle=None) -> pd.DataFrame:
    """Rows with the 30 feature columns -> probability of malignant/benign and a predicted class."""
    bundle = bundle or load_bundle()
    problems = validate_features(df)
    if problems:
        raise ValueError("; ".join(problems))
    thr = float(bundle["threshold"] if threshold is None else threshold)
    X = df[FEATURE_NAMES].to_numpy(dtype=float)
    if bundle["scaler"] is not None:
        X = bundle["scaler"].transform(pd.DataFrame(X, columns=FEATURE_NAMES))
    with torch.no_grad():
        p = torch.sigmoid(bundle["model"](torch.tensor(np.asarray(X), dtype=torch.float32))).numpy().astype(float)

    out = pd.DataFrame({
        "id": df["id"].to_numpy() if "id" in df.columns else np.arange(len(df)),
        "prediction": np.where(p >= thr, "Malignant", "Benign"),
        "p_malignant": np.round(p, 4),
        "p_benign": np.round(1 - p, 4),
    })
    if TARGET in df.columns:
        out["actual"] = np.where(df[TARGET].to_numpy() == "M", "Malignant", "Benign")
        out["correct"] = out["actual"] == out["prediction"]
    return out


def predict_one(features: dict, threshold=None, bundle=None) -> dict:
    row = score_dataframe(pd.DataFrame([features]), threshold, bundle).iloc[0]
    thr = float((bundle or load_bundle())["threshold"] if threshold is None else threshold)
    return {"prediction": str(row["prediction"]), "p_malignant": float(row["p_malignant"]),
            "p_benign": float(row["p_benign"]), "threshold": thr}
