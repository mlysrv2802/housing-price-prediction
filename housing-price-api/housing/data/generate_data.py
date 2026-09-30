"""
generate_data.py
-----------------
In a real company this step doesn't exist — you'd pull data from a warehouse
(Snowflake/BigQuery), a data lake, or an internal API. Since we don't have a
live data source here, this script SIMULATES one: a realistic housing-price
dataset with noise, correlated features, and a categorical column — written
to a CSV, just like an analyst would hand you.

Run this once to create data/housing.csv.
"""

import numpy as np
import pandas as pd

from housing.config import RAW_DATA_PATH, DATA_DIR

RNG = np.random.default_rng(42)


def generate_housing_data(n_rows: int = 2000) -> pd.DataFrame:
    area_sqft = RNG.normal(1500, 500, n_rows).clip(300, 5000)
    bedrooms = RNG.integers(1, 6, n_rows)
    bathrooms = RNG.integers(1, 4, n_rows)
    age_years = RNG.integers(0, 40, n_rows)
    distance_to_city_km = RNG.exponential(8, n_rows).clip(0.5, 60)

    location_tier = RNG.choice(
        ["Tier1", "Tier2", "Tier3"], size=n_rows, p=[0.3, 0.45, 0.25]
    )
    tier_premium = pd.Series(location_tier).map(
        {"Tier1": 40000, "Tier2": 15000, "Tier3": 0}
    ).values

    noise = RNG.normal(0, 15000, n_rows)

    # Ground-truth linear-ish relationship (with a bit of noise) —
    # this is what linear regression will try to recover from the data.
    price = (
        50 * area_sqft
        + 8000 * bedrooms
        + 5000 * bathrooms
        - 800 * age_years
        - 1200 * distance_to_city_km
        + tier_premium
        + 20000
        + noise
    ).clip(min=20000)

    df = pd.DataFrame({
        "area_sqft": area_sqft.round(1),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age_years,
        "distance_to_city_km": distance_to_city_km.round(2),
        "location_tier": location_tier,
        "price": price.round(0),
    })

    # Simulate real-world messiness: a few missing values.
    # Production pipelines must handle this — toy tutorials usually don't.
    missing_idx = RNG.choice(n_rows, size=int(n_rows * 0.02), replace=False)
    df.loc[missing_idx, "bathrooms"] = np.nan

    return df


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_housing_data()
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Wrote {len(df)} rows to {RAW_DATA_PATH}")


if __name__ == "__main__":
    main()
