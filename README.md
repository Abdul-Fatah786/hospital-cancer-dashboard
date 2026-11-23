# 🏥 City Hospital: Live ML Training Dashboard

A PyTorch neural network that classifies breast tumours as **benign or malignant** from 30 cell-nucleus measurements,
plus an interactive **Streamlit dashboard** where you change the training settings and watch the loss curves update live,
and a **FastAPI** service that serves the final model.

> Educational project on the public Breast Cancer Wisconsin (Diagnostic) dataset. **Never use it to diagnose real patients.**

## The question this project answers

> A model can reach almost zero training loss and still perform poorly on unseen data. How can we detect and fix this?

Six experiments compare a baseline, a deliberately overfitting model, and fixes (dropout and weight decay, early stopping,
a smaller model), plus one run without feature scaling. The dashboard classifies each run as **Healthy, Overfitting or Underfitting**.

## Project structure

```
hospital-cancer-dashboard/
├── data/
│   ├── README.md                   # how to get data.csv
│   └── data.csv                    # (you add this)
├── notebooks/
│   └── hospital_experiments.ipynb  # the standalone Google Colab version of the whole project
├── src/
│   ├── config.py                   # paths, feature names, default settings
│   ├── data_loader.py              # load_data(), 60/20/20 split, scaling, tensors
│   ├── model.py                    # HospitalNet, training loop, early stopping, evaluation, diagnose()
│   ├── experiments.py              # the six experiments + run_experiment() + pick_best()
│   ├── evaluate.py                 # plots
│   ├── train.py                    # full pipeline: experiments -> best -> retrain -> test once -> save
│   ├── predict.py                  # score patients with the saved model
│   ├── eda.py                      # exploratory charts
│   └── make_data.py                # optional: build data.csv from scikit-learn if you have no Kaggle account
├── models/                         # hospital_model.pt, scaler.pkl, best_config.json (after training)
├── reports/                        # experiment_history.csv, final_test_metrics.csv, figures/ (after training)
├── api/                            # FastAPI: main.py, schemas.py
├── tests/                          # 31 pytest tests
├── app.py                          # Streamlit dashboard
├── requirements.txt, requirements-dev.txt, pytest.ini, .gitignore
├── requests.http                   # ready-made API calls for the VS Code REST Client
├── Dockerfile, docker-compose.yml
└── .vscode/                        # run/debug configs (F5), settings
```

## Setup in VS Code

1. Install **Python 3.10+** and VS Code with the extensions **Python** and **Jupyter** (optionally **REST Client**).
2. `File → Open Folder` and choose this folder.
3. Open the VS Code terminal and create a virtual environment:

   Windows (PowerShell):
   ```
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements-dev.txt
   ```
   macOS / Linux:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements-dev.txt
   ```
   If PowerShell blocks activation: `Set-ExecutionPolicy -Scope Process RemoteSigned`. PyTorch is a large download, so the install can take a few minutes.
4. `Ctrl+Shift+P` → **Python: Select Interpreter** → choose the `.venv` one.
5. Get the data (see `data/README.md`): download `data.csv` from Kaggle into `data/`, **or** run `python -m src.make_data`.

## Run it (always from the project root)

| Step | Command | Result |
|---|---|---|
| EDA charts | `python -m src.eda` | figures in `reports/figures/` |
| Train | `python -m src.train` | six experiments, final model, test results, reports (about 20 seconds on a laptop CPU) |
| Dashboard | `streamlit run app.py` | http://localhost:8501 |
| API | `uvicorn api.main:app --reload` | http://127.0.0.1:8000/docs |
| Tests | `pytest` | 31 tests pass |

The **Predict** tab and the API need `python -m src.train` to have run once. The **Data explorer, Training lab and Experiments**
tabs work straight away. You can also press **F5** and pick one of the five ready-made run configs.

## Dashboard tabs

- **📊 Data explorer:** class balance, feature distributions by class, correlation with the diagnosis, scale differences.
- **🧪 Training lab:** hidden neurons, dropout, learning rate, weight decay, early-stopping patience, maximum epochs and a scaling switch.
  Press *Train model* and watch the loss curves update live. You get training and validation loss, accuracy, recall, cancers detected and
  missed, false alarms, a confusion matrix, the Healthy / Overfitting / Underfitting diagnosis, and a run history you can download as CSV.
- **🏆 Experiments:** runs the six planned experiments and shows the comparison table, loss curves and metric bars.
- **🔍 Predict:** pick a held-out test patient (actual vs predicted, probabilities) or upload your own CSV.

## How it avoids data leakage

1. The data is split **first** (stratified 60 / 20 / 20: 341 train, 114 validation, 114 test).
2. The scaler is fitted on the **training set only**.
3. Settings are chosen using the **validation** set only.
4. The chosen configuration is retrained on train + validation (scaler refitted on that combined set).
5. The **test set is used once**, at the very end (`python -m src.train`). The dashboard never touches it.

## Diagnosis rules

| Label | Rule |
|---|---|
| Underfitting | final training loss > 0.30 |
| Overfitting | final validation loss > 1.30 × the best validation loss from an **earlier** epoch |
| Healthy | neither |

The last epoch is excluded from the minimum on purpose. Otherwise the final loss could never exceed its own minimum.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | service and model status |
| GET | `/model-info` | chosen configuration and test metrics |
| GET | `/features` | the 30 feature names |
| GET | `/sample?malignant=true` | a held-out example patient |
| POST | `/predict` | score one patient: `{"features": {...30 values...}}` |
| POST | `/predict/batch` | score a list of patients |
| POST | `/predict/csv` | upload a CSV with the 30 feature columns |

`threshold` (0–1) is an optional query parameter on all predict endpoints. Easiest way to try it: open `/docs`, or use
`requests.http` in VS Code. Copy the JSON returned by `/sample` straight into `/predict` (the extra `id` and `actual` keys are ignored).

## Results

Run `python -m src.train`, then fill this in from `reports/final_test_metrics.csv`:

| Metric (test set, evaluated once) | Value |
|---|---|
| Accuracy | |
| Precision | |
| Recall | |
| F1 | |
| Cancers found / missed | |
| False alarms | |

Questions to answer in your report, using `reports/experiment_history.csv` and `reports/figures/loss_curves.png`:

1. Which experiment has the biggest gap between training and validation loss?
2. Did dropout + weight decay (experiment 3) shrink that gap compared with experiment 2?
3. Did early stopping (4) pick a better epoch than running to the end (3)?
4. Did the smaller model (5) generalise better than the baseline?
5. What did removing scaling (6) do to the loss curves?
6. Which experiment missed the fewest cancers? Is it the same one with the best accuracy?

## Notes and limitations

- **Experiment 5** uses 8 neurons. The original plan listed it with the same settings as experiment 1, which would make it identical.
- **Experiment 6 (no scaling)** may be labelled *Overfitting* by the rule because its validation loss is noisy and ends above its early best,
  even though training and validation curves stay close. This is a good point to discuss in your report.
- The dataset is small (569 rows), results come from one random split, and they shift slightly with the random seed.
- Recall matters most here: a false negative is a malignant tumour classified as benign. Accuracy alone is not enough.
- Educational only. Not a medical device.

## Deploy the dashboard (Streamlit Community Cloud)

1. Run `python -m src.train` so `models/` and `data/sample_patients.csv` exist, and commit them (the `.gitignore` already allows this).
2. Put `data/data.csv` in the repo as well, or keep it out and let the app's upload box handle it. The dashboard falls back to an upload.
3. Push to GitHub, then on https://share.streamlit.io choose the repo, branch `main`, main file `app.py`.

The Docker files (`docker-compose up --build`) start the API on port 8000 and the dashboard on port 8501. They are included but were not run here.
