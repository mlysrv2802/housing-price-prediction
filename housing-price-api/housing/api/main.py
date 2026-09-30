"""
api.py
------
Wraps the trained model in a REST API — this is the piece your Angular app
will call over HTTP, the same way it currently calls your Java backend.

Key production ideas here, mapped to what you already know from Java/Spring:

  - Pydantic model (`HouseFeatures`)  ~= a Java DTO/request body class with
    validation annotations. FastAPI validates incoming JSON against it
    automatically and returns a clean 422 error if the client sends garbage
    (missing field, wrong type) — you don't write that validation by hand.
  - Model loaded ONCE at startup (`@app.on_event("startup")`), not per
    request — same principle as a Spring @Bean / singleton service: don't
    reconstruct an expensive object on every call.
  - CORS middleware — required because your Angular dev server (localhost:4200)
    and this API (localhost:8000) are different origins. Without this,
    the browser blocks the request before it even reaches your endpoint.
  - /health endpoint — standard practice so a load balancer / orchestrator
    (k8s, etc.) can check if the service is alive, same idea as Spring
    Actuator's /health.

Run:
    uvicorn api:app --reload --port 8000

Then open http://127.0.0.1:8000/docs — FastAPI auto-generates an interactive
Swagger UI from your pydantic models, so you can test /predict without
writing a single line of frontend code yet.
"""

import logging
import random

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from housing import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Housing Price Predictor API",
    description="Serves predictions from a trained linear regression pipeline.",
    version="1.0.0",
)

# --- CORS ------------------------------------------------------------------
# Angular's dev server runs on http://localhost:4200 by default.
# This API runs on a different port (8000) -> different "origin" in browser
# terms -> blocked by default unless we explicitly allow it here.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # tighten this to your real domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model is loaded once when the process starts, kept in memory, and reused
# for every request — NOT reloaded from disk on each call.
_model = None


@app.on_event("startup")
def load_model_on_startup() -> None:
    global _model
    if not config.PRODUCTION_MODEL_PATH.exists():
        raise RuntimeError(
            f"No production model found at {config.PRODUCTION_MODEL_PATH}. Run `python -m housing.ml.train` first."
        )
    _model = joblib.load(config.PRODUCTION_MODEL_PATH)
    logger.info("Model loaded from %s", config.PRODUCTION_MODEL_PATH)


# --- Request / response schemas --------------------------------------------
# This is your validation layer. FastAPI + pydantic reject bad requests
# automatically, before your logic ever runs.
class HouseFeatures(BaseModel):
    area_sqft: float = Field(..., gt=0, description="Area of the house in square feet")
    bedrooms: int = Field(..., ge=0, le=20)
    bathrooms: float = Field(..., ge=0, le=20)
    age_years: int = Field(..., ge=0, le=200)
    distance_to_city_km: float = Field(..., ge=0)
    location_tier: str = Field(..., description="One of: Tier1, Tier2, Tier3")

    class Config:
        json_schema_extra = {
            "example": {
                "area_sqft": 1400,
                "bedrooms": 3,
                "bathrooms": 2,
                "age_years": 5,
                "distance_to_city_km": 6.2,
                "location_tier": "Tier2",
            }
        }


class PredictionResponse(BaseModel):
    predicted_price: float


@app.get("/health")
def health_check() -> dict:
    """Liveness/readiness check — is the process up AND is the model loaded?"""
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict_price(house: HouseFeatures) -> PredictionResponse:
    if _model is None:
        # Defensive check — should never trigger if startup ran correctly,
        # but production code shouldn't assume happy-path always holds.
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    input_df = pd.DataFrame([house.model_dump()])

    try:
        prediction = _model.predict(input_df)[0]
    except Exception as exc:
        # Never leak raw internals to the client; log the real error server-side.
        logger.exception("Prediction failed for input: %s", house)
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc

    return PredictionResponse(predicted_price=round(float(prediction), 2))


# --- Simulated live data feed -----------------------------------------------
# In a real company, this endpoint would be a THIRD PARTY (a listings vendor,
# a partner API) or an INTERNAL service (your own transactions database
# exposed via API). We don't have a free public housing API to hit, so this
# endpoint stands in for one: each call returns a fresh batch of "new
# listings" that have arrived since your last check, exactly like a real
# vendor feed would.
#
# Differences from a REAL third-party API you should know about:
#   - Real APIs usually require an API key (sent as a header or query param)
#   - Real APIs enforce rate limits (e.g. 100 requests/hour) -- exceeding
#     them returns HTTP 429, and your fetcher needs to back off and retry
#   - Real APIs can go down or time out -- your fetcher must handle that
#     gracefully rather than crashing (see fetch_new_data.py)
@app.get("/new-listings")
def get_new_listings(count: int = 20):
    """Simulates a live feed of newly-sold houses -- INTENTIONALLY messy,
    the way a real third-party API actually behaves. Unlike /predict, this
    endpoint has NO pydantic response_model, so FastAPI won't clean up or
    reject anything here -- exactly like an external vendor's raw JSON
    that you don't control.

    Messiness deliberately injected (with real-world likelihoods):
      - ~5% of records missing a random field entirely
      - ~5% send a number as a STRING ("3" instead of 3)
      - ~5% have inconsistent categorical casing ("tier1", "TIER2")
      - ~3% are exact duplicates of the previous record (vendor re-sent it)
      - ~3% contain an extreme outlier value (bad sensor/data-entry error)
      - ~2% use "N/A" or null for price
      - occasionally an unexpected EXTRA field appears (schema drift)
    """
    listings = []
    previous_record = None

    for _ in range(count):
        area = float(np.clip(random.gauss(1500, 500), 300, 5000))
        bedrooms = random.randint(1, 5)
        bathrooms = float(random.randint(1, 3))
        age = random.randint(0, 40)
        distance = float(np.clip(random.expovariate(1 / 8), 0.5, 60))
        tier = random.choices(["Tier1", "Tier2", "Tier3"], weights=[0.3, 0.45, 0.25])[0]
        tier_premium = {"Tier1": 40000, "Tier2": 15000, "Tier3": 0}[tier]

        price = (
            50 * area + 8000 * bedrooms + 5000 * bathrooms - 800 * age
            - 1200 * distance + tier_premium + 20000 + random.gauss(0, 15000)
        )
        price = max(price, 20000)

        record = {
            "area_sqft": round(area, 1),
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "age_years": age,
            "distance_to_city_km": round(distance, 2),
            "location_tier": tier,
            "price": round(price, 0),
        }

        # --- Inject real-world messiness ------------------------------
        roll = random.random()

        if roll < 0.05:
            # Missing field entirely -- vendor's system dropped it
            field_to_drop = random.choice(list(record.keys()))
            del record[field_to_drop]

        elif roll < 0.10:
            # Number sent as a string -- extremely common with real APIs
            field = random.choice(["area_sqft", "bedrooms", "bathrooms", "age_years"])
            record[field] = str(record[field])

        elif roll < 0.15:
            # Inconsistent categorical casing/formatting
            record["location_tier"] = random.choice(
                [tier.lower(), tier.upper(), f"{tier[:4]} {tier[4:]}"]
            )

        elif roll < 0.18:
            # Extreme outlier / bad sensor value
            bad_field = random.choice(["area_sqft", "bedrooms", "age_years"])
            record[bad_field] = record[bad_field] * 1000

        elif roll < 0.20:
            # Missing/garbage price -- unusable for training
            record["price"] = random.choice([None, "N/A", ""])

        if random.random() < 0.03:
            # Unexpected extra field -- schema drift, e.g. vendor added a
            # new attribute you were never told about
            record["listing_source"] = random.choice(["partner_feed_v2", "walk_in"])

        listings.append(record)

        # Occasionally resend the previous record -- vendor feeds duplicate
        # entries across polls more often than you'd expect
        if previous_record is not None and random.random() < 0.03:
            listings.append(dict(previous_record))

        previous_record = record

    return {"count": len(listings), "listings": listings}
