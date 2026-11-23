"""Central settings: paths, column names, default hyper-parameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "data.csv"                  # Kaggle file (see data/README.md)
SAMPLE_PATH = DATA_DIR / "sample_patients.csv"     # held-out test patients, written by src.train

MODELS_DIR = ROOT / "models"
MODEL_FILE = "hospital_model.pt"
SCALER_FILE = "scaler.pkl"
CONFIG_FILE = "best_config.json"

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

SEED = 42
TARGET = "diagnosis"                               # M = malignant (positive class, 1), B = benign (0)
DROP_COLS = ["id", "Unnamed: 32"]                  # junk columns in the Kaggle file

# 10 measurements x 3 versions (mean, standard error, worst) = 30 features, in the Kaggle column order
BASE_FEATURES = ["radius", "texture", "perimeter", "area", "smoothness", "compactness",
                 "concavity", "concave points", "symmetry", "fractal_dimension"]
FEATURE_NAMES = [f"{b}_{s}" for s in ("mean", "se", "worst") for b in BASE_FEATURES]
MEAN_COLS = [c for c in FEATURE_NAMES if c.endswith("_mean")]

DEFAULT_CFG = dict(hidden=16, dropout=0.3, lr=1e-3, weight_decay=0.05,
                   patience=20, max_epochs=200, scaling=True)    # patience=None -> early stopping OFF
HIDDEN_OPTIONS = [8, 16, 32, 64, 128, 256]
LR_OPTIONS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
