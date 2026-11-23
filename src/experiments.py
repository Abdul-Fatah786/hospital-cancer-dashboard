"""The six experiments from the project plan, and a helper that runs and logs one experiment."""
from datetime import datetime

import pandas as pd

from src.model import diagnose, evaluate_model, make_cfg, train_model


def experiment_plan(max_epochs: int = 200) -> list:
    """(name, config). Each experiment changes as few things as possible compared with the baseline.

    Experiment 5 uses 8 neurons (a genuinely smaller model); with 16 it would be identical to the baseline.
    """
    m = dict(max_epochs=max_epochs)
    return [
        ("1. Baseline",       make_cfg(**m)),
        ("2. Overfitting",    make_cfg(hidden=256, dropout=0.0, weight_decay=0.0, patience=None, **m)),
        ("3. Regularization", make_cfg(hidden=256, dropout=0.3, weight_decay=0.05, patience=None, **m)),
        ("4. Early stopping", make_cfg(hidden=256, dropout=0.3, weight_decay=0.05, patience=20, **m)),
        ("5. Small model",    make_cfg(hidden=8, **m)),
        ("6. No scaling",     make_cfg(scaling=False, **m)),
    ]


def run_experiment(name: str, cfg: dict, prep, progress_cb=None) -> dict:
    """Train on the training set, evaluate on the VALIDATION set. The test set is never touched here."""
    T = prep.tensors[cfg["scaling"]]
    model, hist, best_ep = train_model(cfg, T["Xtr"], prep.ytr, T["Xva"], prep.yva, progress_cb=progress_cb)
    tr = evaluate_model(model, T["Xtr"], prep.ytr)
    va = evaluate_model(model, T["Xva"], prep.yva)
    row = {
        "Experiment ID": name,
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Hidden Neurons": cfg["hidden"],
        "Dropout": cfg["dropout"],
        "Learning Rate": cfg["lr"],
        "Weight Decay": cfg["weight_decay"],
        "Patience": str(cfg["patience"]) if cfg["patience"] is not None else "off",   # always text, so tables stay clean
        "Scaling": cfg["scaling"],
        "Epochs Run": len(hist["train_loss"]),
        "Best Epoch": best_ep,
        "Training Loss": round(tr["loss"], 4),
        "Validation Loss": round(va["loss"], 4),
        "Accuracy": round(va["accuracy"], 4),
        "Precision": round(va["precision"], 4),
        "Recall": round(va["recall"], 4),
        "F1": round(va["f1"], 4),
        "Cancers Found": va["tp"],
        "Cancers Missed": va["fn"],
        "False Alarms": va["fp"],
        "Diagnosis": diagnose(hist),
    }
    return {"row": row, "model": model, "history": hist, "val": va, "cfg": cfg}


def pick_best(results: pd.DataFrame) -> pd.Series:
    """Best experiment by VALIDATION results.

    Runs diagnosed Healthy are preferred (an overfit or underfit run is not a model to ship),
    then the lowest validation loss, then the fewest missed cancers.
    """
    candidates = results[results["Diagnosis"] == "Healthy"]
    if candidates.empty:
        candidates = results
    return candidates.sort_values(["Validation Loss", "Cancers Missed"]).iloc[0]
