"""
config.py
---------
Centralized configuration — the ML-project equivalent of application.properties
or an appsettings.json in a Java/Spring or .NET app.

Never hardcode paths/constants inside your logic files. Import them from here
instead, so changing an environment (dev/staging/prod) means editing ONE file.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent   # project root (housing-price-api/)
STORAGE_DIR = BASE_DIR.parent / "storage"   # goes UP one level, into storage/

DATA_DIR = STORAGE_DIR / "data"
MODEL_DIR = STORAGE_DIR / "models"

RAW_DATA_PATH = DATA_DIR / "housing.csv"


MODEL_PATH = MODEL_DIR / "linear_regression_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

# --- Model versioning / production promotion ------------------------------
# Every training run saves a VERSIONED artifact here (never overwritten).
# Only a version that passes the promotion check gets copied to
# PRODUCTION_MODEL_PATH -- which is the ONLY file predict.py/api.py load.
VERSIONS_DIR = MODEL_DIR / "versions"
PRODUCTION_MODEL_PATH = MODEL_DIR / "production_model.joblib"
PRODUCTION_METRICS_PATH = MODEL_DIR / "production_metrics.json"

# A new model is rejected if its R^2 drops by more than this amount versus
# the current production model.
MAX_ACCEPTABLE_R2_DROP = 0.02

LOG_DIR = BASE_DIR / "logs"
LOG_PATH = LOG_DIR / "training.log"

# --- Data schema ---------------------------------------------------------
# Explicitly naming your columns/types is a production habit: it stops
# silent bugs when someone renames a column upstream in the source data.
TARGET_COLUMN = "price"

NUMERIC_FEATURES = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "age_years",
    "distance_to_city_km",
]

CATEGORICAL_FEATURES = [
    "location_tier",   # e.g. Tier1 / Tier2 / Tier3 city
]

# --- Train/test split & reproducibility ---------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42  # fixed seed -> reproducible experiments, a MUST in production ML

# Ensure directories exist as soon as config is imported, so any module
# (train.py, predict.py, logging setup) can rely on these paths existing.
for _dir in (DATA_DIR, MODEL_DIR, LOG_DIR, VERSIONS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
