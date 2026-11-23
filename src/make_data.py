"""Optional: create data/data.csv from the copy of this dataset that ships inside scikit-learn.

Use it ONLY if you cannot download the file from Kaggle. It is the same Breast Cancer Wisconsin (Diagnostic)
data (569 rows, 30 features, 212 malignant / 357 benign), written in the Kaggle column format.
The `id` column is synthetic because scikit-learn does not include the original patient ids.

    python -m src.make_data            # writes data/data.csv (refuses to overwrite)
    python -m src.make_data --force    # overwrite
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from src.config import DATA_PATH, FEATURE_NAMES, TARGET


def build_dataframe() -> pd.DataFrame:
    d = load_breast_cancer()
    cols = []
    for n in d.feature_names:
        if n.startswith("mean "):
            base, suffix = n[5:], "_mean"
        elif n.startswith("worst "):
            base, suffix = n[6:], "_worst"
        else:                                        # "radius error" etc.
            base, suffix = n[: -len(" error")], "_se"
        base = base if base == "concave points" else base.replace(" ", "_")
        cols.append(base + suffix)
    df = pd.DataFrame(d.data, columns=cols)[FEATURE_NAMES]
    df.insert(0, TARGET, np.where(d.target == 0, "M", "B"))   # in scikit-learn, 0 = malignant
    df.insert(0, "id", np.arange(900001, 900001 + len(df)))
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if DATA_PATH.exists() and not args.force:
        raise SystemExit(f"{DATA_PATH} already exists. Use --force to overwrite.")
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = build_dataframe()
    df.to_csv(DATA_PATH, index=False)
    print(f"Wrote {DATA_PATH}  {df.shape}  {df[TARGET].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
