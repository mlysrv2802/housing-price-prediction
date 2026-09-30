"""
fetch_new_data.py
-------------------
Pulls fresh data from a LIVE source (here: our own /new-listings endpoint,
standing in for a real vendor/internal API) and appends it to housing.csv.

This is the piece you'd schedule to run periodically in production (cron,
Airflow, a Kubernetes CronJob) -- NOT something a person runs by hand
every time. Run it manually for now to see it work.

Real-world differences to keep in mind (noted inline below):
  - A real third-party API needs an API key/auth header
  - A real API can rate-limit you (HTTP 429) -- handle it, don't hammer it
  - A real API call can time out or fail -- never let that crash a
    scheduled job silently; log it and exit non-zero so monitoring notices

Run: python -m housing.data.fetch_new_data
"""

import logging
import sys

import pandas as pd
import requests

from housing import config
import os


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT_SECONDS = 10  # NEVER call an external API with no timeout --
                                # a hung request can block a scheduled job forever


def fetch_new_listings(count: int = 20) -> list[dict]:
    """Calls the live endpoint. Returns the new records, or raises on failure
    (the caller decides what to do -- crash loudly vs. skip this run)."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/new-listings",
            params={"count": count},
            timeout=REQUEST_TIMEOUT_SECONDS,
            # A real third-party API would need something like:
            # headers={"Authorization": f"Bearer {API_KEY}"},
        )
        response.raise_for_status()  # raises for 4xx/5xx status codes
    except requests.exceptions.Timeout:
        logger.error("Request to %s timed out after %ss", API_BASE_URL, REQUEST_TIMEOUT_SECONDS)
        raise
    except requests.exceptions.ConnectionError:
        logger.error("Could not connect to %s -- is the API running?", API_BASE_URL)
        raise
    except requests.exceptions.HTTPError as exc:
        # A real API returning 429 (rate limited) would need a backoff-and-retry
        # here instead of just failing. Keeping it simple for now.
        logger.error("API returned an error status: %s", exc)
        raise

    payload = response.json()
    listings = payload.get("listings", [])
    logger.info("Fetched %d new listings from %s", len(listings), API_BASE_URL)
    return listings


def validate_and_clean(records: list[dict]) -> pd.DataFrame:
    """Never trust incoming data blindly -- this is where a real ML pipeline
    earns its keep. Each step below handles a specific real-world data
    quality problem, logged so you can see exactly what got fixed/dropped
    and why (essential for debugging a production pipeline later)."""
    df = pd.DataFrame(records)
    original_count = len(df)

    # --- 1. Unexpected extra fields (schema drift) ------------------------
    # Don't let an unrecognized column silently ride along into training --
    # drop it, but log it so you notice if a vendor adds something you
    # actually want to make use of later.
    expected_columns = set(config.NUMERIC_FEATURES) | set(config.CATEGORICAL_FEATURES) | {config.TARGET_COLUMN}
    extra_columns = set(df.columns) - expected_columns
    if extra_columns:
        logger.info("Dropping unexpected columns not in our schema: %s", extra_columns)
        df = df.drop(columns=list(extra_columns))

    # --- 2. Missing columns entirely ---------------------------------------
    # If a required column is missing from EVERY record, that's a schema
    # problem worth failing loudly on -- something upstream fundamentally
    # changed. If it's just missing on SOME rows, pandas already fills
    # those cells with NaN, handled naturally below.
    missing_entirely = expected_columns - set(df.columns)
    if missing_entirely:
        raise ValueError(f"API response is missing required columns entirely: {missing_entirely}")

    # --- 3. Numbers sent as strings -----------------------------------------
    # Coerce to numeric; anything that truly can't be parsed becomes NaN
    # instead of silently corrupting downstream math.
    numeric_columns = config.NUMERIC_FEATURES + [config.TARGET_COLUMN]
    for col in numeric_columns:
        before_numeric = df[col].apply(lambda v: isinstance(v, (int, float))).sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")  # "N/A", "", None -> NaN
        after_numeric = df[col].notna().sum()
        if before_numeric != len(df):
            logger.info("Column '%s': coerced string/garbage values to numeric (or NaN)", col)

    # --- 4. Inconsistent categorical casing/formatting ---------------------
    # "tier1", "TIER2", "Tier 3" all need to normalize to "Tier1"/"Tier2"/"Tier3"
    # before they hit the model's OneHotEncoder -- otherwise each variant
    # becomes its OWN separate (wrong) category.
    def normalize_tier(value) -> str:
        if not isinstance(value, str):
            return value
        cleaned = value.replace(" ", "").strip().lower()
        mapping = {"tier1": "Tier1", "tier2": "Tier2", "tier3": "Tier3"}
        return mapping.get(cleaned, value)  # leave unrecognized values as-is -> caught by validation below

    before_values = df["location_tier"].copy()
    df["location_tier"] = df["location_tier"].apply(normalize_tier)
    n_normalized = (before_values != df["location_tier"]).sum()
    if n_normalized:
        logger.info("Normalized casing/formatting on %d location_tier values", n_normalized)

    # --- 5. Drop rows with missing target (unusable for training) ----------
    before = len(df)
    df = df.dropna(subset=[config.TARGET_COLUMN])
    if len(df) < before:
        logger.warning("Dropped %d rows with missing/unparseable price", before - len(df))

    # --- 6. Drop rows with missing critical features ------------------------
    before = len(df)
    df = df.dropna(subset=["area_sqft"])  # area is too important to impute at ingestion time
    if len(df) < before:
        logger.warning("Dropped %d rows with missing area_sqft", before - len(df))

    # --- 7. Outlier / sanity bounds ------------------------------------------
    # Bounds based on what's physically plausible for a house -- reject
    # obvious data errors (a sensor/typo producing area_sqft=1,609,400).
    before = len(df)
    df = df[df["area_sqft"].between(100, 20000)]
    df = df[df["bedrooms"].between(0, 20) | df["bedrooms"].isna()]
    df = df[df["age_years"].between(0, 300) | df["age_years"].isna()]
    if len(df) < before:
        logger.warning("Dropped %d rows with out-of-range outlier values", before - len(df))

    # --- 8. Unrecognized categorical values (after normalization) -----------
    before = len(df)
    df = df[df["location_tier"].isin(["Tier1", "Tier2", "Tier3"])]
    if len(df) < before:
        logger.warning("Dropped %d rows with unrecognized location_tier values", before - len(df))

    # --- 9. Exact duplicate records ------------------------------------------
    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        logger.warning("Dropped %d exact duplicate records", before - len(df))

    logger.info(
        "Validation: %d raw records in -> %d clean records out (%.0f%% kept)",
        original_count, len(df), 100 * len(df) / original_count if original_count else 0,
    )
    return df


def append_to_dataset(new_df: pd.DataFrame) -> int:
    """Returns the number of rows actually appended (0 if nothing to add) --
    the scheduler uses this return value to decide whether a retrain is
    worth triggering."""
    if new_df.empty:
        logger.info("No valid new rows to append.")
        return 0

    existing_df = pd.read_csv(config.RAW_DATA_PATH) if config.RAW_DATA_PATH.exists() else pd.DataFrame()
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.to_csv(config.RAW_DATA_PATH, index=False)

    logger.info(
        "Appended %d new rows. Dataset size: %d -> %d",
        len(new_df), len(existing_df), len(combined_df),
    )
    return len(new_df)


def main() -> int:
    """Returns the number of new rows appended (0 on a clean no-op run).
    Exits the process with status 1 on failure, for standalone/cron use."""
    try:
        raw_listings = fetch_new_listings(count=1000)
    except requests.exceptions.RequestException:
        logger.error("Fetch failed -- exiting without modifying the dataset.")
        sys.exit(1)  # non-zero exit so a scheduler (cron/Airflow) flags this run as failed

    if not raw_listings:
        logger.info("API returned zero new listings this run.")
        return 0

    clean_df = validate_and_clean(raw_listings)
    return append_to_dataset(clean_df)


if __name__ == "__main__":
    main()