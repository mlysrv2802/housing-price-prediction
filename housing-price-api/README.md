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
housing_price_predictor/
├── config.py          # paths, feature lists, constants
├── generate_data.py   # simulates a real data export (run once)
├── data_loader.py      # loads + validates the dataset (schema checks)
├── train.py            # builds pipeline, trains, evaluates, saves model
├── predict.py           # loads saved model, scores new records
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── data/ models/ logs/  (created automatically)
```

## Run it
```bash
pip install -r requirements.txt

python -m housing.data.generate_data # creates data/housing.csv
python -m housing.ml.train           # trains + saves model + models/metrics.json
python predict.py         # loads the saved model, predicts on sample houses
pytest tests/             # runs the test suite
```

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

## What would change for a REST API in production
Wrap `predict()` from `predict.py` in a FastAPI/Flask endpoint (or, since
you're an Angular/Java dev, think Spring Boot controller calling a Python
microservice). The pipeline object is already serialized — the endpoint
just loads it once at startup and calls `.predict()` per request.

## Natural next steps
1. Swap `LinearRegression` for `Ridge`/`Lasso` (regularization) — same
   pipeline, one line changed.
2. Add cross-validation (`cross_val_score`) instead of a single train/test
   split, for a more robust metric.
3. Track experiments with MLflow instead of a single `metrics.json`.
4. Add a `/predict` FastAPI endpoint + Dockerfile for real deployment.
