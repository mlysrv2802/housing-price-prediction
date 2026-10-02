# Housing Price Predictor — Linear Regression (Production-Style)

A small, realistic ML project structured the way it would look in a real
company codebase — not a single Jupyter cell. Built for a backend dev
learning ML: every ML concept below is mapped to something you already know.

## Why this looks different from tutorial code

| Tutorial notebook | This project |
|---|---|
| `df.dropna()` inline, model in the same cell | `Pipeline` handles missing values + encoding, reusable at train AND inference |
| Hardcoded file paths | `config.py` — one source of truth, like `application.properties` |
| `print(score)` | `logging` to file + console, metrics saved to `metrics.json` |
| Model lives only in memory | Model serialized with `joblib` — load it in a service without retraining |
| No tests | `tests/` — the ML equivalent of unit tests, meant to run in CI |

## Project structure
```
housing-price-api/
├── housing/
│   ├── config.py               # paths, feature lists, constants
│   ├── api/
│   │   └── main.py             # FastAPI app: /predict, /health, /new-listings
│   ├── data/
│   │   ├── generate_data.py    # simulates a real data export (run once)
│   │   ├── data_loader.py      # loads + validates the dataset (schema checks)
│   │   └── fetch_new_data.py   # pulls fresh records from /new-listings
│   ├── ml/
│   │   ├── train.py            # builds pipeline, trains, evaluates, saves candidate model
│   │   ├── model_registry.py   # versioning, promotion, rollback
│   │   └── predict.py          # loads saved model, scores sample records
│   └── jobs/
│       └── scheduler.py        # fetch -> retrain -> promote-if-better, on a timer
├── experiments/                # one-off analysis and tuning scripts
├── tests/
│   └── test_pipeline.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Run it
Run everything from this folder (`housing-price-api/`).

```bash
pip install -r requirements.txt

python -m housing.data.generate_data # creates the dataset (housing.csv)
python -m housing.ml.train           # trains + saves model + metrics
python -m housing.ml.predict         # loads the saved model, predicts on sample houses
pytest tests/                        # runs the test suite
```

Data and models are stored in `../storage/` (`data/`, `models/`), one level
above this folder, as set in `housing/config.py`.

### Run the experiments
These are optional, one-off scripts for analysis and model selection. They
don't affect the production model.

```bash
python -m experiments.eda_feature_analysis    # run first on any new dataset
python -m experiments.check_linearity         # is a linear model appropriate?
python -m experiments.compare_models          # LinearRegression vs Ridge vs Lasso vs RandomForest
python -m experiments.tune_hyperparameters    # GridSearchCV for Ridge/Lasso alpha
python -m experiments.tune_randomforest       # RandomizedSearchCV for RandomForest
python -m experiments.scratch_gradient_descent  # linear regression in pure numpy
```

### Run the API
```bash
uvicorn housing.api.main:app --reload --port 8000
```
Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.

| Endpoint | Purpose |
|---|---|
| `POST /predict` | predicted price for one house (validated by pydantic) |
| `GET /health` | liveness check, reports whether the model is loaded |
| `GET /new-listings?count=20` | simulated, deliberately messy vendor feed |

### Keep the model fresh
```bash
python -m housing.data.fetch_new_data  # one-off pull from /new-listings (API must be running)
python -m housing.jobs.scheduler       # long-running: fetch -> retrain -> promote if not worse
```

### Docker
```bash
docker compose up -d          # bootstrap data/model, then start the API
docker compose up -d --build  # after code changes
docker compose down
```

## What each Python script does

### `housing/`
| Script | Purpose |
|---|---|
| `config.py` | Single source of truth: paths (`../storage/...`), feature lists, target column. Like `application.properties`. |
| `data/generate_data.py` | Simulates a data export by writing a realistic synthetic housing CSV. Run once. |
| `data/data_loader.py` | Loads the raw CSV and runs basic schema/sanity checks. Keeps data access separate from training. |
| `data/fetch_new_data.py` | Pulls fresh records from the `/new-listings` feed, handles failures, and appends them to the dataset. |
| `ml/train.py` | Builds the preprocessing + LinearRegression pipeline, trains, evaluates (RMSE/MAE/R²), and saves a candidate model with its metrics. |
| `ml/model_registry.py` | Versions each trained model, promotes it only if it isn't meaningfully worse than production, and supports rollback. |
| `ml/predict.py` | Loads the production model and scores sample houses from the command line. |
| `api/main.py` | FastAPI service exposing `/predict`, `/health` and `/new-listings`. Loads the model once at startup. |
| `jobs/scheduler.py` | Long-running loop: fetch new data, retrain if any arrived, promote the new model only if it's not worse. |

### `experiments/`
| Script | Purpose |
|---|---|
| `eda_feature_analysis.py` | Exploratory analysis: missing values, correlation with price, multicollinearity (VIF), and RandomForest feature importance. |
| `check_linearity.py` | Checks whether linear regression suits the data. Saves feature-vs-price scatter plots and residual diagnostics to `experiments/outputs/`. |
| `compare_models.py` | Trains LinearRegression, Ridge, Lasso and RandomForest on the same split and compares their metrics. |
| `tune_hyperparameters.py` | Uses cross-validated `GridSearchCV` to find the best `alpha` for Ridge and Lasso. |
| `tune_randomforest.py` | Uses `RandomizedSearchCV` to tune several RandomForest hyperparameters together. |
| `scratch_gradient_descent.py` | Linear regression with gradient descent in plain numpy, to show what `.fit()` hides. Reads `data/housing.csv` relative to the current folder. |

### `tests/`
| Script | Purpose |
|---|---|
| `test_pipeline.py` | pytest suite for the pipeline and evaluation, meant to run in CI. |

## The ML concepts, mapped to what you already know

- **Pipeline** (`sklearn.pipeline.Pipeline`) — like chaining middleware in
  Express/Spring: each step (impute → scale/encode → model) transforms the
  data and passes it on. Fit it once on training data; call `.predict()` on
  new data and it re-applies the exact same steps. This is what prevents
  "it worked in my notebook but broke in prod" bugs.
- **ColumnTransformer** — routes different columns to different
  preprocessing, similar to routing different request fields to different
  validators.
- **train_test_split** — you never evaluate a model on data it trained on
  (that's like a dev writing their own passing test for buggy code). We
  hold out 20% as a test set purely to check generalization.
- **StandardScaler / OneHotEncoder** — linear regression's coefficients are
  just weights in `y = w1*x1 + w2*x2 + ... + b`. Numbers need to be on
  comparable scales, and category strings ("Tier1") need to become numeric
  columns (0/1 flags) — the model literally cannot use raw text.
- **RMSE / MAE / R²** — RMSE and MAE tell you the average prediction error
  in the same units as your target (₹ here); R² tells you what fraction of
  the variance in price the model explains (1.0 = perfect, 0 = no better
  than guessing the average).
- **Coefficients as explainability** — after training, you can literally
  read off "+1 more bedroom → price goes up by X" from `regressor.coef_`.
  This is the #1 reason linear regression survives in industry: stakeholders
  can question and audit it, unlike a black-box deep net.
- **joblib.dump/load** — model persistence, same idea as serializing an
  object to disk, so a separate process (an API server) can load it without
  needing your training code or raw data.

## How the REST API works
`housing/api/main.py` is a FastAPI app (think Spring Boot controller calling
a Python microservice). The pipeline object is already serialized — the app
loads it once at startup and calls `.predict()` per request.

## Natural next steps
1. Swap `LinearRegression` for `Ridge`/`Lasso` (regularization) — same
   pipeline, one line changed.
2. Add cross-validation (`cross_val_score`) instead of a single train/test
   split, for a more robust metric.
3. Track experiments with MLflow instead of a single `metrics.json`.
4. Add API-key auth and rate limiting to the API before real deployment.
