"""
data_loader.py
---------------
Single responsibility: load raw data and do basic sanity checks.
Keeping this separate from training logic means you can swap the data
source (CSV -> SQL query -> API call) without touching train.py at all —
same idea as an interface/DAO layer in a Java backend.
"""

import logging

import pandas as pd

from housing.config import RAW_DATA_PATH, TARGET_COLUMN, NUMERIC_FEATURES, CATEGORICAL_FEATURES

logger = logging.getLogger(__name__)


def load_raw_data() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"No data found at {RAW_DATA_PATH}. Run `python -m housing.data.generate_data` first."
        )

    df = pd.read_csv(RAW_DATA_PATH)
    logger.info("Loaded %d rows, %d columns from %s", len(df), df.shape[1], RAW_DATA_PATH)

    _validate_schema(df)
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Fail fast and loudly if the data doesn't match what the model expects.

    This is the ML equivalent of validating a request DTO before it hits
    your service layer. Silent schema drift is one of the most common
    causes of production ML bugs.
    """
    required_columns = set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES) | {TARGET_COLUMN}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing values — cannot train on these rows.")

    if len(df) < 50:
        logger.warning("Dataset has only %d rows — model quality may be poor.", len(df))
