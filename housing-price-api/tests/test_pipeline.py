"""
test_pipeline.py
-----------------
A minimal but realistic test suite. In a production ML repo, tests like
these run in CI (GitHub Actions / Jenkins) on every pull request — the
same instinct you already have from JUnit/Jasmine testing, applied to ML.

Run: pytest tests/
"""

import pandas as pd
import pytest

from housing import config
from housing.ml.train import build_pipeline, evaluate


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "area_sqft": [1000, 1500, 2000, 1200, 1800],
        "bedrooms": [2, 3, 4, 2, 3],
        "bathrooms": [1, 2, 2, 1, 2],
        "age_years": [10, 5, 1, 20, 3],
        "distance_to_city_km": [5.0, 3.0, 1.0, 15.0, 2.0],
        "location_tier": ["Tier2", "Tier1", "Tier1", "Tier3", "Tier2"],
        "price": [60000, 120000, 180000, 40000, 130000],
    })


def test_pipeline_trains_and_predicts_without_error(sample_df):
    X = sample_df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
    y = sample_df[config.TARGET_COLUMN]

    pipeline = build_pipeline()
    pipeline.fit(X, y)
    preds = pipeline.predict(X)

    assert len(preds) == len(y)
    assert all(p > 0 for p in preds), "Predicted prices should be positive"


def test_pipeline_handles_missing_values(sample_df):
    sample_df.loc[0, "bathrooms"] = None  # simulate a missing value
    X = sample_df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
    y = sample_df[config.TARGET_COLUMN]

    pipeline = build_pipeline()
    pipeline.fit(X, y)  # should not raise, thanks to SimpleImputer
    preds = pipeline.predict(X)
    assert len(preds) == len(y)


def test_pipeline_handles_unseen_category(sample_df):
    """A category never seen during training must not crash inference —
    this happens ALL the time in production (new city, new product type)."""
    X_train = sample_df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
    y_train = sample_df[config.TARGET_COLUMN]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    new_row = X_train.iloc[[0]].copy()
    new_row["location_tier"] = "Tier99_never_seen"
    preds = pipeline.predict(new_row)  # handle_unknown="ignore" prevents a crash
    assert len(preds) == 1


def test_evaluate_returns_expected_keys():
    import numpy as np
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 305])
    metrics = evaluate(y_true, y_pred)
    assert set(metrics.keys()) == {"rmse", "mae", "r2_score"}
