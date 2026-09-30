# Housing Price Prediction

An end-to-end ML project: a scikit-learn linear regression model served by a FastAPI backend, with an Angular form as the frontend and a scheduler that periodically pulls new data and retrains.

```
housing-price-prediction/
├── housing-price-api/   # Python: training, model registry, REST API, scheduler
├── housing-price-ui/    # Angular app that calls the API
└── storage/             # data/ and models/ (created automatically, shared by API + scheduler)
```

## How it fits together

```
 Angular UI ──POST /predict──▶ FastAPI (api/main.py) ──loads──▶ storage/models/production_model.joblib
                                   ▲                                   ▲
                                   │ GET /new-listings                 │ promoted only if not worse
                              scheduler ─▶ fetch_new_data ─▶ train ─▶ model_registry
```

1. `housing/ml/train.py` builds a sklearn `Pipeline` (impute → scale/one-hot → `LinearRegression`), evaluates it (RMSE, MAE, R²) and saves a versioned model.
2. `housing/ml/model_registry.py` promotes a new version to `production_model.joblib` only if its R² doesn't drop by more than `MAX_ACCEPTABLE_R2_DROP` (0.02, see `housing/config.py`).
3. `housing/api/main.py` loads the production model once at startup and serves predictions.
4. `housing/jobs/scheduler.py` bootstraps a model if none exists, then repeatedly fetches new listings from `/new-listings` (a deliberately messy simulated vendor feed), validates and appends them to the dataset, retrains, and lets the registry decide whether to promote.

## Quick start (local)

Requires Python 3.12 and Node 22.

**API**
```bash
cd housing-price-api
pip install -r requirements.txt

python -m housing.data.generate_data       # creates storage/data/housing.csv
python -m housing.ml.train                 # trains, saves + promotes a model
uvicorn housing.api.main:app --reload --port 8000   # Swagger UI at http://127.0.0.1:8000/docs
```

**UI**
```bash
cd housing-price-ui
npm install
ng serve                                   # http://localhost:4200
```

**Scheduler (optional)** — with the API running: `python -m housing.jobs.scheduler`

## Quick start (Docker)

From the repo root (needs Docker Compose v2.20+):
```bash
docker compose up -d --build     # api :8000, scheduler, ui :4200
docker compose logs -f
docker compose down
```

The root [docker-compose.yml](docker-compose.yml) `include`s [housing-price-api/docker-compose.yml](housing-price-api/docker-compose.yml) (api + scheduler) and adds the UI service (multi-stage Node build → nginx). You can still run the API stack alone from `housing-price-api/`.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check; reports whether the model is loaded |
| POST | `/predict` | Predict a price from house features |
| GET | `/new-listings?count=20` | Simulated third-party feed of newly sold houses (intentionally messy) |

Example:
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "area_sqft": 1400, "bedrooms": 3, "bathrooms": 2,
  "age_years": 5, "distance_to_city_km": 6.2, "location_tier": "Tier2"
}'
# {"predicted_price": ...}
```

`location_tier` is one of `Tier1`, `Tier2`, `Tier3`. Invalid input returns a 422.

## Scripts in `housing-price-api/`

```
housing/                  # the runtime package (this is what ships in the Docker image)
├── config.py             # paths, feature lists, split/seed, promotion threshold
├── api/main.py           # FastAPI app: /health, /predict, /new-listings
├── ml/
│   ├── train.py          # builds, trains, evaluates and saves the pipeline
│   ├── predict.py        # scores sample houses with the saved model
│   └── model_registry.py # versioning, promotion and rollback
├── data/
│   ├── generate_data.py  # generates the initial synthetic dataset
│   ├── data_loader.py    # loads and schema-validates the dataset
│   └── fetch_new_data.py # pulls, cleans, appends new listings (API_BASE_URL env var)
└── jobs/scheduler.py     # bootstrap + fetch → retrain → promote loop (APScheduler)
experiments/              # EDA, tuning, model comparison; not shipped in the image
└── outputs/              # generated plots (PNGs)
tests/
```

Run modules from `housing-price-api/` with `python -m`, e.g. `python -m housing.ml.model_registry list` or `python -m experiments.compare_models`.

## Tests

```bash
cd housing-price-api
pytest tests/
```

The UI uses Vitest: `cd housing-price-ui && ng test`.

## Configuration notes

- The API URL is hardcoded in [prediction.service.ts](housing-price-ui/src/app/services/prediction.service.ts) as `http://127.0.0.1:8000`; move it to an environment file for real deployments.
- CORS in [main.py](housing-price-api/housing/api/main.py) only allows `http://localhost:4200`. If you serve the UI from Docker on another origin, add it there.
- Models and data live in `storage/` next to `housing-price-api/`, not inside it.

## More detail

- [housing-price-api/README.md](housing-price-api/README.md): ML concepts explained for backend developers.
- [housing-price-ui/README.md](housing-price-ui/README.md): Angular CLI commands.
