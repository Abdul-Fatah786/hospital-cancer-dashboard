"""Loading the data, splitting it 60/20/20 and preparing tensors."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import DATA_PATH, FEATURE_NAMES, SEED, TARGET


def load_data(source=DATA_PATH) -> pd.DataFrame:
    """Read data.csv (a path or an uploaded file) and check that it looks right."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {path}.\n"
                "Download data.csv from https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data "
                "and put it in the data/ folder (see data/README.md),\n"
                "or run:  python -m src.make_data   (same data from scikit-learn)."
            )
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(source)

    missing = [c for c in FEATURE_NAMES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"The file is missing expected columns: {missing}")
    bad = set(df[TARGET].dropna().unique()) - {"M", "B"}
    if bad or df[TARGET].isnull().any():
        raise ValueError(f"'{TARGET}' must contain only M or B (found: {sorted(map(str, bad)) or 'missing values'})")
    if df[FEATURE_NAMES].isnull().any().any():
        raise ValueError("The feature columns contain missing values")
    if "id" not in df.columns:
        df.insert(0, "id", np.arange(len(df)))
    return df.reset_index(drop=True)


def split_data(df: pd.DataFrame, seed: int = SEED) -> dict:
    """Stratified 60 / 20 / 20 split. Malignant = 1 (positive class), benign = 0."""
    X = df[FEATURE_NAMES]
    y = (df[TARGET] == "M").astype(int)
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=seed)
    return dict(X_train=X_train, X_val=X_val, X_test=X_test, X_dev=X_dev,
                y_train=y_train, y_val=y_val, y_test=y_test, y_dev=y_dev,
                ids_test=df.loc[X_test.index, "id"])


def to_tensor(a) -> torch.Tensor:
    return torch.tensor(np.asarray(a), dtype=torch.float32)


@dataclass
class Prepared:
    """Everything the experiments need.  tensors[True] = scaled features, tensors[False] = raw features."""
    split: dict
    scaler: StandardScaler
    tensors: dict
    ytr: torch.Tensor
    yva: torch.Tensor
    yte: torch.Tensor


def prepare(df: pd.DataFrame, seed: int = SEED) -> Prepared:
    sp = split_data(df, seed)
    scaler = StandardScaler().fit(sp["X_train"])          # fit on TRAIN only (prevents data leakage)
    tensors = {
        True: dict(Xtr=to_tensor(scaler.transform(sp["X_train"])), Xva=to_tensor(scaler.transform(sp["X_val"])),
                   Xte=to_tensor(scaler.transform(sp["X_test"]))),
        False: dict(Xtr=to_tensor(sp["X_train"].values), Xva=to_tensor(sp["X_val"].values),
                    Xte=to_tensor(sp["X_test"].values)),
    }
    return Prepared(sp, scaler, tensors, to_tensor(sp["y_train"]), to_tensor(sp["y_val"]), to_tensor(sp["y_test"]))
