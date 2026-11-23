import pytest

from src.make_data import build_dataframe


@pytest.fixture(scope="session")
def df():
    """The real 569-row dataset (bundled with scikit-learn), in Kaggle format. No internet needed."""
    return build_dataframe()


@pytest.fixture(scope="session")
def prep(df):
    from src.data_loader import prepare
    return prepare(df)


@pytest.fixture(scope="session")
def trained(tmp_path_factory, df):
    """Run the whole pipeline once into a temporary folder (capped at 80 epochs to keep tests quick)."""
    from src.train import train_pipeline
    root = tmp_path_factory.mktemp("project")
    out = train_pipeline(df, max_epochs=80, models_dir=root / "models", reports_dir=root / "reports",
                         data_dir=root / "data", verbose=False)
    out["root"] = root
    return out
