"""
train.py
--------
Production-style training script for a Linear Regression model.

Key ideas that separate this from "tutorial" code:
  1. Preprocessing lives INSIDE a sklearn Pipeline, not as loose pandas calls.
     -> The exact same code path runs at training AND at inference time,
        so you can never get train/serve skew (a very common real bug).
  2. Missing values and categorical encoding are handled explicitly.
  3. The model is evaluated with multiple metrics, and metrics are saved
     to disk (metrics.json) so you can track model quality over time —
     the ML equivalent of a build's test-coverage report.
  4. The trained pipeline is serialized with joblib so `predict.py` (or an
     API server) can load it without retraining.
  5. Logging instead of print() — because in production you ship logs,
     not stdout.

Run: python -m housing.ml.train
"""

import json
import logging

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from housing import config
from housing.ml import model_registry
from housing.data.data_loader import load_raw_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(config.LOG_PATH)],
)
logger = logging.getLogger(__name__)


def build_pipeline() -> Pipeline:
    """Preprocessing + model, bundled as ONE object.

    ColumnTransformer lets numeric and categorical columns get different
    treatment in a single, reusable step:
      - numeric  -> fill missing with median, then scale to mean 0 / std 1
      - categorical -> one-hot encode (turn "Tier1"/"Tier2"/"Tier3" into
        separate 0/1 columns, since linear regression only understands numbers)

    Scaling matters for linear regression when you regularize (Ridge/Lasso)
    or compare coefficient magnitudes; it's included here as standard practice.
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, config.NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, config.CATEGORICAL_FEATURES),
    ])

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])
    return model


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2_score": r2}


def main() -> bool:
    """Returns True if this run's model was promoted to production,
    False if it was rejected (production model left unchanged)."""
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw_data()

    X = df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
    y = df[config.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    logger.info("Train size: %d | Test size: %d", len(X_train), len(X_test))

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = evaluate(y_test.values, y_pred)
    logger.info("Test metrics: %s", metrics)

    # Save this run as a permanent, versioned artifact FIRST -- it's never
    # lost even if it gets rejected below. Then decide whether it earns
    # promotion to production (the file predict.py/api.py actually load).
    version_id = model_registry.save_version(pipeline, metrics)
    promoted = model_registry.promote_if_better(version_id, metrics)

    if promoted:
        logger.info("Version %s is now the production model.", version_id)
    else:
        logger.warning(
            "Version %s was REJECTED -- production model unchanged. "
            "Run `python -m housing.ml.model_registry list` to review all versions.",
            version_id,
        )

    # Legacy paths kept in sync too, for any older script/notebook that
    # still points at config.MODEL_PATH directly.
    joblib.dump(pipeline, config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    logger.info("Saved model to %s", config.MODEL_PATH)
    logger.info("Saved metrics to %s", config.METRICS_PATH)

    # Inspect learned coefficients — one of linear regression's biggest
    # practical advantages: you can explain the model to a non-ML stakeholder.
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefficients = pipeline.named_steps["regressor"].coef_
    coef_table = pd.Series(coefficients, index=feature_names).sort_values(
        key=abs, ascending=False
    )
    logger.info("Top feature coefficients:\n%s", coef_table.to_string())

    return promoted


if __name__ == "__main__":
    main()
