import io

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_NAMES
from src.data_loader import load_data, split_data


def test_dataset_shape_and_classes(df):
    assert df.shape == (569, 32)                      # id + diagnosis + 30 features
    assert df["diagnosis"].value_counts().to_dict() == {"B": 357, "M": 212}
    assert len(FEATURE_NAMES) == 30 and list(df[FEATURE_NAMES].columns) == FEATURE_NAMES


def test_split_sizes_match_the_plan(df):
    sp = split_data(df)
    assert (len(sp["y_train"]), len(sp["y_val"]), len(sp["y_test"])) == (341, 114, 114)


def test_splits_are_disjoint_and_stratified(df):
    sp = split_data(df)
    tr, va, te = set(sp["X_train"].index), set(sp["X_val"].index), set(sp["X_test"].index)
    assert not (tr & va) and not (tr & te) and not (va & te)
    rates = [sp[k].mean() for k in ("y_train", "y_val", "y_test")]
    assert max(rates) - min(rates) < 0.02             # malignant share is about 37% in every split


def test_malignant_is_the_positive_class(df):
    sp = split_data(df)
    idx = sp["y_train"].index[0]
    assert sp["y_train"].loc[idx] == (1 if df.loc[idx, "diagnosis"] == "M" else 0)


def test_scaler_is_fitted_on_training_data_only(prep):
    assert np.allclose(prep.scaler.mean_, prep.split["X_train"].mean().to_numpy())
    # scaled training data has mean ~0 and std ~1, validation data does not (it was not used to fit)
    assert abs(prep.tensors[True]["Xtr"].mean().item()) < 1e-4
    assert not np.allclose(prep.tensors[True]["Xva"].mean(0).numpy(), 0, atol=1e-6)


def test_raw_tensors_are_unscaled(prep):
    assert prep.tensors[False]["Xtr"].max().item() > 100      # area_worst is in the thousands


def test_load_data_cleans_junk_and_validates(df, tmp_path):
    p = tmp_path / "data.csv"
    df.assign(**{"Unnamed: 32": np.nan}).to_csv(p, index=False)
    assert load_data(p).shape[0] == 569
    with pytest.raises(FileNotFoundError, match="make_data"):
        load_data(tmp_path / "nope.csv")
    with pytest.raises(ValueError, match="missing expected columns"):
        load_data(io.StringIO(df.drop(columns=["radius_mean"]).to_csv(index=False)))
    with pytest.raises(ValueError, match="only M or B"):
        load_data(io.StringIO(df.assign(diagnosis="X").to_csv(index=False)))
    with pytest.raises(ValueError, match="missing values"):
        bad = df.copy(); bad.loc[0, "area_mean"] = np.nan
        load_data(io.StringIO(bad.to_csv(index=False)))
    no_id = load_data(io.StringIO(df.drop(columns=["id"]).to_csv(index=False)))
    assert "id" in no_id.columns
