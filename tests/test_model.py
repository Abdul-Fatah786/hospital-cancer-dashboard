import numpy as np

from src.experiments import experiment_plan, pick_best, run_experiment
from src.model import diagnose, evaluate_model, make_cfg, train_model


def test_diagnose_underfitting():
    assert diagnose({"train_loss": [0.7, 0.5, 0.45], "val_loss": [0.7, 0.5, 0.45]}) == "Underfitting"


def test_diagnose_overfitting_uses_earlier_best_epoch():
    h = {"train_loss": [0.20, 0.08, 0.02], "val_loss": [0.20, 0.10, 0.50]}
    assert diagnose(h) == "Overfitting"               # 0.50 > 1.3 x 0.10


def test_diagnose_healthy_and_single_epoch():
    assert diagnose({"train_loss": [0.2, 0.1], "val_loss": [0.2, 0.19]}) == "Healthy"
    assert diagnose({"train_loss": [0.1], "val_loss": [0.5]}) == "Healthy"   # nothing earlier to compare with


def test_training_reduces_the_loss(prep):
    T = prep.tensors[True]
    _, hist, _ = train_model(make_cfg(max_epochs=30), T["Xtr"], prep.ytr, T["Xva"], prep.yva)
    assert hist["train_loss"][-1] < hist["train_loss"][0]
    assert len(hist["train_loss"]) == len(hist["val_loss"]) == 30


def test_training_is_reproducible(prep):
    T = prep.tensors[True]
    _, h1, _ = train_model(make_cfg(max_epochs=10), T["Xtr"], prep.ytr, T["Xva"], prep.yva)
    _, h2, _ = train_model(make_cfg(max_epochs=10), T["Xtr"], prep.ytr, T["Xva"], prep.yva)
    assert np.allclose(h1["val_loss"], h2["val_loss"])


def test_early_stopping_stops_and_restores_best_weights(prep):
    T = prep.tensors[True]
    cfg = make_cfg(hidden=256, dropout=0.0, weight_decay=0.0, patience=5, max_epochs=200)
    model, hist, best_ep = train_model(cfg, T["Xtr"], prep.ytr, T["Xva"], prep.yva)
    assert len(hist["val_loss"]) < 200                              # stopped early
    assert abs(min(hist["val_loss"]) - hist["val_loss"][best_ep - 1]) < 1e-9
    assert abs(evaluate_model(model, T["Xva"], prep.yva)["loss"] - min(hist["val_loss"])) < 1e-4   # best weights restored


def test_progress_callback_is_called_every_epoch(prep):
    T = prep.tensors[True]
    calls = []
    train_model(make_cfg(max_epochs=7, patience=None), T["Xtr"], prep.ytr, T["Xva"], prep.yva,
                progress_cb=lambda e, h, m: calls.append((e, m)))
    assert calls == [(i, 7) for i in range(1, 8)]


def test_evaluate_counts_add_up(prep):
    T = prep.tensors[True]
    model, _, _ = train_model(make_cfg(max_epochs=40), T["Xtr"], prep.ytr, T["Xva"], prep.yva)
    ev = evaluate_model(model, T["Xva"], prep.yva)
    assert ev["tp"] + ev["fn"] == int(prep.yva.sum())
    assert ev["tp"] + ev["fn"] + ev["fp"] + ev["tn"] == len(prep.yva)
    assert ev["accuracy"] > 0.9 and ev["recall"] > 0.85


def test_experiment_plan_changes_the_right_things():
    plan = dict(experiment_plan())
    assert len(plan) == 6
    assert plan["2. Overfitting"]["hidden"] == 256 and plan["2. Overfitting"]["patience"] is None
    assert plan["2. Overfitting"]["dropout"] == 0 and plan["2. Overfitting"]["weight_decay"] == 0
    assert plan["4. Early stopping"]["patience"] == 20
    assert plan["5. Small model"]["hidden"] < plan["1. Baseline"]["hidden"]
    assert plan["6. No scaling"]["scaling"] is False


def test_run_experiment_and_pick_best_prefers_healthy(prep):
    import pandas as pd
    rows = [run_experiment(n, c, prep)["row"] for n, c in experiment_plan(max_epochs=40)]
    res = pd.DataFrame(rows)
    assert {"Experiment ID", "Training Loss", "Validation Loss", "Cancers Found", "Diagnosis"} <= set(res.columns)
    assert pick_best(res)["Diagnosis"] == "Healthy"
    fake = pd.DataFrame([{"Experiment ID": "a", "Diagnosis": "Overfitting", "Validation Loss": 0.1, "Cancers Missed": 0},
                         {"Experiment ID": "b", "Diagnosis": "Healthy", "Validation Loss": 0.2, "Cancers Missed": 3}])
    assert pick_best(fake)["Experiment ID"] == "b"
