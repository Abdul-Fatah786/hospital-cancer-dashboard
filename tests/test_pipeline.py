import json

import pandas as pd

from src.config import FEATURE_NAMES
from src.predict import load_bundle, predict_one, score_dataframe


def test_pipeline_writes_all_artifacts(trained):
    root = trained["root"]
    for f in ("models/hospital_model.pt", "models/best_config.json", "reports/experiment_history.csv",
              "reports/final_test_metrics.csv", "reports/test_report.txt", "data/sample_patients.csv",
              "reports/figures/loss_curves.png", "reports/figures/metrics_comparison.png",
              "reports/figures/confusion_matrix_test.png"):
        assert (root / f).exists(), f
    if trained["best_cfg"]["scaling"]:
        assert (root / "models" / "scaler.pkl").exists()


def test_six_experiments_logged_with_all_columns(trained):
    res = trained["results"]
    assert len(res) == 6
    assert {"Experiment ID", "Hidden Neurons", "Dropout", "Learning Rate", "Weight Decay", "Patience", "Scaling",
            "Epochs Run", "Best Epoch", "Training Loss", "Validation Loss", "Accuracy", "Precision", "Recall", "F1",
            "Cancers Found", "Cancers Missed", "False Alarms", "Diagnosis"} <= set(res.columns)


def test_big_unregularised_model_overfits_more(trained):
    res = trained["results"].set_index("Experiment ID")
    gap = lambda n: res.loc[n, "Validation Loss"] - res.loc[n, "Training Loss"]
    assert gap("2. Overfitting") > gap("3. Regularization")


def test_final_model_is_good_on_the_test_set(trained):
    t = trained["test"]
    assert t["accuracy"] > 0.93 and t["recall"] > 0.88
    assert t["tp"] + t["fn"] == 42 and t["tp"] + t["fn"] + t["fp"] + t["tn"] == 114   # the 114 held-out patients


def test_saved_config_matches_the_chosen_experiment(trained):
    cfg = json.loads((trained["root"] / "models" / "best_config.json").read_text())
    assert cfg["experiment"] == trained["best_name"]
    assert trained["results"].set_index("Experiment ID").loc[trained["best_name"], "Diagnosis"] == "Healthy"


def test_saved_model_reproduces_the_test_result(trained):
    bundle = load_bundle(trained["root"] / "models")
    sample = pd.read_csv(trained["root"] / "data" / "sample_patients.csv")
    scored = score_dataframe(sample, bundle=bundle)
    assert len(scored) == 114
    assert int(scored["correct"].sum()) == trained["test"]["tp"] + trained["test"]["tn"]   # same answers as in training


def test_predict_one_and_threshold(trained):
    bundle = load_bundle(trained["root"] / "models")
    sample = pd.read_csv(trained["root"] / "data" / "sample_patients.csv")
    mal = sample[sample["diagnosis"] == "M"].iloc[0][FEATURE_NAMES].to_dict()
    r = predict_one(mal, bundle=bundle)
    assert set(r) == {"prediction", "p_malignant", "p_benign", "threshold"}
    assert abs(r["p_malignant"] + r["p_benign"] - 1) < 1e-3
    ben = sample[sample["diagnosis"] == "B"].iloc[0][FEATURE_NAMES].to_dict()
    assert predict_one(ben, threshold=0.0, bundle=bundle)["prediction"] == "Malignant"   # threshold 0 flags everything
    assert predict_one(ben, threshold=1.0, bundle=bundle)["prediction"] == "Benign"      # threshold 1 flags nothing
