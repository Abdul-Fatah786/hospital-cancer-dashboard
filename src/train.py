"""Run the six experiments, pick the best on VALIDATION, retrain on train+validation, test ONCE, save everything.

Usage (from the project root):
    python -m src.train
    python -m src.train --max-epochs 100
    python -m src.train --data path/to/data.csv
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from src.config import (CONFIG_FILE, DATA_DIR, DATA_PATH, FEATURE_NAMES, MODEL_FILE, MODELS_DIR, REPORTS_DIR,
                        SCALER_FILE, TARGET)
from src.data_loader import load_data, prepare, to_tensor
from src.evaluate import confusion_figure, loss_curve_grid, metrics_bar_figure, save_fig
from src.experiments import experiment_plan, pick_best, run_experiment
from src.model import evaluate_model, train_model


def train_pipeline(df: pd.DataFrame, max_epochs: int = 200, models_dir=MODELS_DIR, reports_dir=REPORTS_DIR,
                   data_dir=DATA_DIR, verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    models_dir, reports_dir, data_dir = Path(models_dir), Path(reports_dir), Path(data_dir)
    fig_dir = reports_dir / "figures"
    for d in (models_dir, fig_dir, data_dir):
        d.mkdir(parents=True, exist_ok=True)

    prep = prepare(df)
    sp = prep.split
    log(f"Split: train={len(sp['y_train'])} val={len(sp['y_val'])} test={len(sp['y_test'])} | "
        f"malignant in train/val/test = {int(sp['y_train'].sum())}/{int(sp['y_val'].sum())}/{int(sp['y_test'].sum())}")

    # 1) six experiments, evaluated on the VALIDATION set
    rows, histories, cfgs = [], {}, {}
    for name, cfg in experiment_plan(max_epochs):
        out = run_experiment(name, cfg, prep)
        rows.append(out["row"]); histories[name] = out["history"]; cfgs[name] = cfg
        r = out["row"]
        log(f"  {name:<20} epochs={r['Epochs Run']:>3} best={r['Best Epoch']:>3} train={r['Training Loss']:.3f} "
            f"val={r['Validation Loss']:.3f} recall={r['Recall']:.3f} missed={r['Cancers Missed']} -> {r['Diagnosis']}")
    results = pd.DataFrame(rows)

    # 2) best configuration by validation results only
    best = pick_best(results)
    best_name, best_cfg = best["Experiment ID"], cfgs[best["Experiment ID"]]
    log(f"\nBest configuration (validation): {best_name}")

    # 3) retrain on train + validation (scaler refitted on that combined set)
    if best_cfg["scaling"]:
        final_scaler = StandardScaler().fit(sp["X_dev"])
        Xdev, Xte = to_tensor(final_scaler.transform(sp["X_dev"])), to_tensor(final_scaler.transform(sp["X_test"]))
    else:
        final_scaler = None
        Xdev, Xte = to_tensor(sp["X_dev"].values), to_tensor(sp["X_test"].values)
    # With early stopping the best epoch is the right length; without it the run used all its epochs.
    final_epochs = int(best["Epochs Run"]) if best_cfg["patience"] is None else max(int(best["Best Epoch"]), 1)
    final_model, _, _ = train_model(best_cfg, Xdev, to_tensor(sp["y_dev"]), fixed_epochs=final_epochs)

    # 4) the ONE look at the test set
    test = evaluate_model(final_model, Xte, prep.yte)
    log(f"Retrained on {len(sp['y_dev'])} samples for {final_epochs} epochs.")
    log(f"TEST: accuracy={test['accuracy']:.3f} precision={test['precision']:.3f} recall={test['recall']:.3f} "
        f"f1={test['f1']:.3f} | cancers found {test['tp']}/{test['tp'] + test['fn']}, missed {test['fn']}, false alarms {test['fp']}")

    # 5) save the model (plain tensors + numbers only, so it loads with weights_only=True)
    test_metrics = {k: float(v) for k, v in test.items() if k not in ("probs", "preds")}
    torch.save({"state_dict": final_model.state_dict(), "n_features": len(FEATURE_NAMES),
                "hidden": best_cfg["hidden"], "dropout": best_cfg["dropout"], "feature_names": list(FEATURE_NAMES),
                "scaling": bool(best_cfg["scaling"]), "threshold": 0.5, "experiment": str(best_name),
                "final_epochs": final_epochs, "test_metrics": test_metrics}, models_dir / MODEL_FILE)
    if final_scaler is not None:
        joblib.dump(final_scaler, models_dir / SCALER_FILE)
    elif (models_dir / SCALER_FILE).exists():
        (models_dir / SCALER_FILE).unlink()          # no stale scaler next to an unscaled model
    (models_dir / CONFIG_FILE).write_text(json.dumps(
        {"experiment": best_name, "config": best_cfg, "final_epochs": final_epochs}, indent=2))

    # 6) tables and figures
    results.to_csv(reports_dir / "experiment_history.csv", index=False)
    pd.DataFrame([test_metrics]).to_csv(reports_dir / "final_test_metrics.csv", index=False)
    (reports_dir / "test_report.txt").write_text(classification_report(
        sp["y_test"], test["preds"], target_names=["Benign", "Malignant"], digits=3))
    save_fig(loss_curve_grid(histories, dict(zip(results["Experiment ID"], results["Diagnosis"]))), fig_dir / "loss_curves.png")
    save_fig(metrics_bar_figure(results), fig_dir / "metrics_comparison.png")
    save_fig(confusion_figure(confusion_matrix(sp["y_test"], test["preds"], labels=[0, 1]), "Confusion matrix (test)"),
             fig_dir / "confusion_matrix_test.png")

    # 7) the held-out test patients, used by the dashboard and the API as ready-made examples
    sample = df.loc[sp["X_test"].index, ["id", TARGET] + FEATURE_NAMES]
    sample.to_csv(data_dir / "sample_patients.csv", index=False)

    log(f"\nSaved model + scaler in {models_dir}, tables and figures in {reports_dir}")
    return {"results": results, "best_name": best_name, "best_cfg": best_cfg, "test": test,
            "test_metrics": test_metrics, "final_epochs": final_epochs, "histories": histories}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the City Hospital model")
    ap.add_argument("--data", default=str(DATA_PATH), help="path to data.csv")
    ap.add_argument("--max-epochs", type=int, default=200)
    args = ap.parse_args(argv)
    try:
        df = load_data(args.data)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))
    print(f"Loaded {len(df)} patients, {int((df[TARGET] == 'M').sum())} malignant")
    train_pipeline(df, max_epochs=args.max_epochs)


if __name__ == "__main__":
    main()
