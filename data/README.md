# Data

Put the Kaggle file here as **`data/data.csv`**.

Download: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data  (free Kaggle account needed)

Or with the Kaggle CLI:

    pip install kaggle
    kaggle datasets download -d uciml/breast-cancer-wisconsin-data -p data --unzip

**No Kaggle account?** Run `python -m src.make_data`. It writes `data/data.csv` from the copy of the same
dataset that ships inside scikit-learn (same 569 rows and 30 features; the `id` column is synthetic).

Facts: 569 patients, 357 benign (B) and 212 malignant (M), 30 numeric features (10 measurements x mean / se / worst),
target column `diagnosis`. The Kaggle file also has an empty `Unnamed: 32` column, which is ignored.

`sample_patients.csv` is created by `python -m src.train`. It holds the 114 held-out test patients and is used by the
dashboard's Predict tab and the API's `/sample` endpoint.
