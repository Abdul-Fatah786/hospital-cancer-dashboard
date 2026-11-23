"""FastAPI service.   Run (from the project root):  uvicorn api.main:app --reload
Interactive docs: http://127.0.0.1:8000/docs
"""
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from api.schemas import BatchRequest, BatchResponse, HealthResponse, Patient, Prediction
from src.config import FEATURE_NAMES, SAMPLE_PATH
from src.predict import load_bundle, score_dataframe

MAX_CSV_ROWS = 20_000

app = FastAPI(
    title="City Hospital Tumour Classifier API",
    version="1.0.0",
    description="Educational project (Breast Cancer Wisconsin dataset). NEVER use it to diagnose real patients.",
)


def _bundle():
    try:
        return load_bundle()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


def _rows(out: pd.DataFrame) -> list:
    return [{"id": int(r.id), "prediction": r.prediction, "p_malignant": float(r.p_malignant),
             "p_benign": float(r.p_benign)} for r in out.itertuples()]


@app.get("/", tags=["info"])
def root():
    return {"service": "City Hospital Tumour Classifier API", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse, tags=["info"])
def health():
    try:
        load_bundle()
        return HealthResponse(status="ok", model_loaded=True)
    except FileNotFoundError as e:
        return HealthResponse(status="model missing", model_loaded=False, detail=str(e))


@app.get("/model-info", tags=["info"])
def model_info():
    m = _bundle()["meta"]
    return {"experiment": m["experiment"], "hidden": m["hidden"], "dropout": m["dropout"], "scaling": m["scaling"],
            "final_epochs": m["final_epochs"], "threshold": m["threshold"], "test_metrics": m["test_metrics"]}


@app.get("/features", tags=["info"], summary="The 30 feature names the model expects")
def features():
    return {"n_features": len(FEATURE_NAMES), "features": FEATURE_NAMES}


@app.get("/sample", tags=["info"], summary="A held-out example patient you can paste into /predict")
def sample(malignant: bool = Query(False, description="Return a malignant example instead of a benign one")):
    if not SAMPLE_PATH.exists():
        raise HTTPException(404, "data/sample_patients.csv not found. Run: python -m src.train")
    df = pd.read_csv(SAMPLE_PATH)
    pool = df[df["diagnosis"] == ("M" if malignant else "B")]
    if pool.empty:
        raise HTTPException(404, "No example of that class in the sample file")
    row = pool.iloc[0]
    return {"id": int(row["id"]), "actual": "Malignant" if malignant else "Benign",
            "features": {f: float(row[f]) for f in FEATURE_NAMES}}


@app.post("/predict", response_model=Prediction, tags=["predict"])
def predict(patient: Patient, threshold: float = Query(None, ge=0, le=1, description="Override the saved threshold (0.5)")):
    b = _bundle()
    out = score_dataframe(pd.DataFrame([patient.features]), threshold, b)
    thr = float(b["threshold"] if threshold is None else threshold)
    r = out.iloc[0]
    return Prediction(prediction=r["prediction"], p_malignant=float(r["p_malignant"]),
                      p_benign=float(r["p_benign"]), threshold=thr)


@app.post("/predict/batch", response_model=BatchResponse, tags=["predict"])
def predict_batch(req: BatchRequest, threshold: float = Query(None, ge=0, le=1)):
    b = _bundle()
    out = score_dataframe(pd.DataFrame([p.features for p in req.patients]), threshold, b)
    thr = float(b["threshold"] if threshold is None else threshold)
    return BatchResponse(n_patients=len(out), n_malignant=int((out["prediction"] == "Malignant").sum()),
                         threshold=thr, results=_rows(out))


@app.post("/predict/csv", response_model=BatchResponse, tags=["predict"], summary="Upload a CSV with the 30 feature columns")
def predict_csv(file: UploadFile = File(...), threshold: float = Query(None, ge=0, le=1)):
    b = _bundle()
    try:
        df = pd.read_csv(file.file)
    except Exception:
        raise HTTPException(400, "Could not read the upload as a CSV file")
    if len(df) > MAX_CSV_ROWS:
        raise HTTPException(413, f"Too many rows ({len(df):,}); the limit is {MAX_CSV_ROWS:,}")
    try:
        out = score_dataframe(df, threshold, b)
    except ValueError as e:
        raise HTTPException(422, str(e))
    thr = float(b["threshold"] if threshold is None else threshold)
    return BatchResponse(n_patients=len(out), n_malignant=int((out["prediction"] == "Malignant").sum()),
                         threshold=thr, results=_rows(out))
