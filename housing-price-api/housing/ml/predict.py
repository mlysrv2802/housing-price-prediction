"""
predict.py
----------
Loads the persisted pipeline and scores NEW data — this is the piece that
would sit behind a REST endpoint in production (e.g. wrapped in a FastAPI
/predict route called by your Angular frontend, analogous to any other
microservice call).

Run: python -m housing.ml.predict
"""

import logging

import joblib
import pandas as pd

from housing import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_model():
    if not config.PRODUCTION_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No production model at {config.PRODUCTION_MODEL_PATH}. Run `python -m housing.ml.train` first."
        )
    return joblib.load(config.PRODUCTION_MODEL_PATH)


def predict(records: list[dict]) -> list[float]:
    """
    records: list of dicts shaped like the training features, e.g.
        {"area_sqft": 1400, "bedrooms": 3, "bathrooms": 2,
         "age_years": 5, "distance_to_city_km": 6.2, "location_tier": "Tier2"}

    Because the SAME pipeline object does imputation + scaling + encoding,
    you never have to remember "which columns did I scale again?" at
    inference time — that class of bug is eliminated by construction.
    """
    model = load_model()
    df = pd.DataFrame(records)
    preds = model.predict(df)
    return preds.tolist()


if __name__ == "__main__":
    sample_houses = [
        {
            "area_sqft": 1400, "bedrooms": 3, "bathrooms": 2,
            "age_years": 5, "distance_to_city_km": 6.2, "location_tier": "Tier2",
        },
        {
            "area_sqft": 2500, "bedrooms": 4, "bathrooms": 3,
            "age_years": 1, "distance_to_city_km": 2.0, "location_tier": "Tier1",
        },
        {
            "area_sqft": 800, "bedrooms": 1, "bathrooms": 1,
            "age_years": 25, "distance_to_city_km": 30.0, "location_tier": "Tier3",
        },
    ]

    predictions = predict(sample_houses)
    for house, price in zip(sample_houses, predictions):
        logger.info("Predicted price for %s: ₹%.0f", house, price)
